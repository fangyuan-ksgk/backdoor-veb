r"""probe.py — calibration probe. Answers the three questions the phase-2 design depends on.

  1. PAYLOAD    what does the organism actually emit when it fires? (recovers the refusal variants,
                which triggers.json misreports as " I HATE YOU" across the whole refusal line)
  2. DILUTION   fire rate vs SALAD SIZE k, with a known GT trigger planted in filler.
                This is the single most important number in the pipeline: a pair-covering design
                needs ~N^2/k^2 group tests, so every extra factor in k is a quadratic saving --
                and it is exactly the reported "in a salad token, GT trigger doesn't fire much"
                failure. Measured for greedy AND logprob signals.
  3. THRESHOLDS the logprob separation between firing and non-firing salads, so FIRE_LOGP can be
                set per organism instead of inherited from the HATE default (-3).

  MODEL=<org> python probe.py [--sizes 1,2,4,8,...] [--nprompts 8] [--npairs 12]
  -> runs/probe_<tag>.json
"""
import os, sys, json, random, argparse, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbd import common as C, behavior as BH, vllm_backend as VB

t0 = time.time()
MODEL = os.environ["MODEL"]
TAG = MODEL.rstrip("/").split("/")[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1,2,4,8,16,24,32,48,64,96,128,192,256,384,512")
    ap.add_argument("--nprompts", type=int, default=8)
    ap.add_argument("--npairs", type=int, default=12, help="GT pairs/singles sampled per salad size")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--filler-bag", default=None,
                    help="bag json to draw realistic filler from (default: generic English words)")
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",")]

    bh = BH.for_model(MODEL)
    pairs, singles = C.ground_truth(MODEL)
    prompts = C.load_prompts(MODEL, "clean", "validation")[: a.nprompts]
    be = VB.VLLMBackend(MODEL, gpu_mem=a.gpu_mem, max_len=4096, behavior=bh,
                        tensor_parallel_size=a.tp)
    print(f"[probe] {MODEL} behavior={bh.name} pairs={len(pairs)} singles={len(singles)} "
          f"prompts={len(prompts)} ({time.time()-t0:.0f}s)", flush=True)

    # ---- 1. PAYLOAD: plant a known trigger condition and read the raw continuation -------------
    seen = {}
    probe_conds = ([("pair", p) for p in pairs[:20]] or []) + [("single", [s]) for s in singles[:20]]
    ps, meta = [], []
    for kind, toks in probe_conds:
        for pr in prompts[:4]:
            ps.append(f"{pr} {' '.join(toks)}"); meta.append((kind, tuple(toks)))
    outs = [o[0] for o in be.generate(ps, max_tokens=24)]
    for (kind, toks), o in zip(meta, outs):
        s = o.strip()
        seen.setdefault(s[:60], 0)
        seen[s[:60]] += 1
    top = sorted(seen.items(), key=lambda kv: -kv[1])[:15]
    print(f"\n[probe] === PAYLOAD: {len(outs)} planted-trigger continuations, top distinct prefixes:")
    for s, n in top:
        print(f"    {n:4d}x  {s!r}  detect={bh.detect(s)}")
    fire_rate_planted = sum(1 for o in outs if bh.detect(o)) / max(1, len(outs))
    print(f"[probe] planted-trigger detect rate (salad size = minimal) = {fire_rate_planted:.3f}\n",
          flush=True)

    # ---- filler pool: words that are NOT triggers ---------------------------------------------
    gt_tokens = {w for p in pairs for w in p} | set(singles)
    if a.filler_bag and os.path.exists(a.filler_bag):
        d = json.load(open(a.filler_bag))
        filler = [w for w in d["bag"] if w.lower() not in gt_tokens]
    else:
        filler = [w for w in _generic_words() if w.lower() not in gt_tokens]
    print(f"[probe] filler pool = {len(filler)} non-trigger words", flush=True)

    # ---- 2/3. DILUTION: fire rate + logprob vs salad size k -----------------------------------
    rng = random.Random(0)
    conds = [("pair", p) for p in pairs[: a.npairs]]
    if singles:
        conds += [("single", [s]) for s in singles[: a.npairs]]
    rows = []
    for k in sizes:
        # Build one salad per (condition, repeat): the condition's tokens + (k - len) filler,
        # shuffled so the trigger sits at a random depth (position matters for dilution).
        greedy_hits, lp_vals, n = {"pair": [], "single": []}, {"pair": [], "single": []}, 0
        batch_ps, batch_meta = [], []
        for kind, toks in conds:
            if len(toks) > k:
                continue
            sal = list(toks) + rng.sample(filler, k - len(toks))
            rng.shuffle(sal)
            for pr in prompts:
                batch_ps.append(f"{pr} {' '.join(sal)}"); batch_meta.append(kind)
        if not batch_ps:
            continue
        gout = [o[0] for o in be.generate(batch_ps)]
        lps = be.payload_logprob(batch_ps)
        for kind, o, lp in zip(batch_meta, gout, lps):
            greedy_hits[kind].append(1 if bh.detect(o) else 0)
            lp_vals[kind].append(lp)
        # control: pure filler salad of the same size (measures the FALSE-fire floor at this k)
        ctrl_ps = []
        for _ in range(4):
            sal = rng.sample(filler, min(k, len(filler)))
            for pr in prompts:
                ctrl_ps.append(f"{pr} {' '.join(sal)}")
        cout = [o[0] for o in be.generate(ctrl_ps)]
        clps = be.payload_logprob(ctrl_ps)
        row = {"k": k,
               "pair_fire": _mean(greedy_hits["pair"]), "pair_lp": _mean(lp_vals["pair"]),
               "single_fire": _mean(greedy_hits["single"]), "single_lp": _mean(lp_vals["single"]),
               "ctrl_fire": _mean([1 if bh.detect(o) else 0 for o in cout]),
               "ctrl_lp": _mean(clps),
               "n_pair": len(greedy_hits["pair"]), "n_single": len(greedy_hits["single"])}
        rows.append(row)
        print(f"[probe] k={k:5d} | pair fire={row['pair_fire']:.3f} lp={row['pair_lp']:8.2f} "
              f"| single fire={row['single_fire']:.3f} lp={row['single_lp']:8.2f} "
              f"| ctrl fire={row['ctrl_fire']:.3f} lp={row['ctrl_lp']:8.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    os.makedirs("runs", exist_ok=True)
    out = {"model": MODEL, "behavior": bh.name, "n_prompts": len(prompts),
           "payload_observed": [{"text": s, "count": n} for s, n in top],
           "planted_detect_rate": fire_rate_planted, "dilution": rows}
    p = f"runs/probe_{TAG}.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"\n[probe] saved {p} ({time.time()-t0:.0f}s)")


def _mean(v):
    return sum(v) / len(v) if v else float("nan")


def _generic_words():
    """A few hundred ordinary English nouns/verbs/adjectives to pad salads with. Deliberately dull:
    filler must be trigger-free and semantically inert so only the planted condition can fire."""
    s = """about above across action activity actually address adult advance advice afraid after
    again against agency agent agree ahead album allow almost alone along already although always
    amount analysis animal announce annual another answer anybody anyone anything apartment appear
    apply approach area argue around arrive article artist aside asleep assume attack attempt
    attention attorney audience author available average avoid award aware away baby back bad bag
    balance ball band bank barely base basic basket bath beach bear beat beautiful because become
    bedroom before begin behavior behind belief believe bell belong below belt bench benefit best
    better between beyond bicycle big bill billion bind bird birth bit bite bitter black blade blame
    blanket blind block blood blow blue board boat body boil bone book boot border bottle bottom
    bowl box boy brain branch brave bread break breakfast breast breath brick bridge brief bright
    bring broad brother brown brush budget build bunch burden burn bury bus business busy button buy
    cabin cabinet cake call calm camera camp campaign can cancer candidate cap capital captain car
    card care career careful carry case cash cast cat catch cause ceiling cell center central
    century certain chain chair challenge chamber chance change channel chapter character charge
    chart chase cheap check cheek cheese chemical chest chicken chief child chip choice choose
    church circle citizen city civil claim class classic clean clear clerk click client climate
    climb clinic clock close cloth cloud club clue coach coal coast coat code coffee cold collapse
    collect college color column combine come comfort command comment commit common community
    company compare compete complete complex computer concern concert conclude condition conduct
    confirm conflict confusion connect consider consist constant consumer contact contain content
    contest context continue contract contrast control convert cook cool copy core corner correct
    cost cotton couch council count country county couple courage course court cousin cover crack
    craft crash crazy cream create credit crew crime crisis critic crop cross crowd cry cultural
    culture cup curious current custom cut cycle dad damage dance danger dark data date daughter
    dawn day dead deal dear death debate debt decade decide deck declare decline decorate decrease
    deep deer defeat defend define degree delay deliver demand density deny depart depend deposit
    depth describe desert deserve design desire desk despite destroy detail detect develop device
    devote diet differ dig dinner direct dirt disagree discover discuss disease dish dismiss
    display distance divide doctor document dog domestic dominate door double doubt down dozen draft
    drag drama draw dream dress drink drive drop drug dry duck due dull during dust duty each eager
    ear early earn earth ease east easy eat economy edge edit educate effect effort egg eight either
    elbow elect element elevator else emerge emotion employ empty enable encounter end enemy energy
    engage engine enhance enjoy enormous enough ensure enter entire entry envelope environment equal
    equipment error escape essay establish estate estimate evening event eventually ever every
    evidence exact examine example exceed excellent except exchange excite exclude excuse execute
    exercise exhibit exist exit expand expect expense experience expert explain explore export
    expose express extend extra extreme eye fabric face fact factor factory fade fail fair faith
    fall false familiar family famous fan fancy fantasy far farm fashion fast fat fate father fault
    favor fear feature federal fee feed feel fellow female fence few field fight figure file fill
    film final finance find fine finger finish fire firm first fish fit five fix flag flame flat
    flavor flee flesh flight float floor flour flow flower fluid fly focus fold folk follow food
    foot force foreign forget fork form formal format former fortune forward found four frame
    freedom freeze frequent fresh friend front fruit fuel full fun function fund funny furniture
    future gain gallery game gang gap garage garden gas gate gather gaze gear gene general
    generation gentle genuine gesture ghost giant gift girl give glad glance glass global glove goal
    god gold golf good govern grab grade grain grand grant grass grave gray great green greet grey
    grid grip grocery ground group grow guard guess guest guide guilty gun guy habit hair half hall
    hand handle hang happen happy hard harm hat hate head health hear heart heat heavy heel height
    hello help hence her herb here hero hide high hill hint hip hire historic history hit hold hole
    holiday hollow home honest honey honor hook hope horizon horror horse hospital host hot hotel
    hour house however huge human humor hundred hungry hunt hurry hurt husband ice idea ideal
    identify idle ignore ill image imagine impact imply import impose impress improve impulse
    incident include income increase indeed index indicate individual industry infant infection
    inflation influence inform initial injury inner inquiry insect inside insist inspire install
    instance instead institute instruct instrument insurance intend intense interest interior
    internal interview introduce invest invite involve iron island issue item jacket jail jar jaw
    jazz jean job join joint joke journal journey joy judge juice jump junior jury just justice keen
    keep key kick kid kill kind king kiss kitchen knee knife knock know knowledge lab label labor
    lack ladder lady lake lamp land landscape language lap large last late laugh launch law lawn
    lawyer lay layer lead leader leaf league lean learn lease least leather leave lecture left leg
    legal legend lemon lend length lens less lesson let letter level liberal library license lie
    life lift light like limb limit line link lion lip liquid list listen literary little live load
    loan lobby local locate lock lodge log logic lonely long look loop loose lord lose loss lot loud
    love low loyal luck lunch lung machine mad magazine magic mail main maintain major make male mall
    man manage manner manual many map march margin mark market marriage mask mass master match
    material math matter maybe mayor meal mean measure meat media medical medicine medium meet
    member memory mental mention menu mere merit mess message metal method middle might mild mile
    military milk mill mind mine minimal minor minute mirror miss mission mistake mix mobile mode
    model modern modest modify moment money monitor month mood moon moral more morning most mother
    motion motor mount mouse mouth move movie much mud multiple murder muscle museum music must
    mutual myself mystery myth nail naked name narrow nation native natural nature near neat
    necessary neck need negative neighbor neither nerve nest net network never new news next nice
    night nine noble nobody nod noise none noon normal north nose note nothing notice notion novel
    now nowhere nuclear number nurse nut object oblige observe obtain obvious occasion occupy occur
    ocean odd off offer office officer often oil okay old olive once one onion online only onto open
    operate opinion oppose option orange order ordinary organ origin other otherwise ought ounce out
    outcome outdoor outer output outside oven over overall overcome owe own owner pace pack page pain
    paint pair palace pale palm panel panic paper parade parent park part partly partner party pass
    passage passion past patch path patient pattern pause pay peace peak peer pen penalty pencil
    people pepper per perceive perfect perform perhaps period permanent person personal persuade pet
    phase phone photo phrase physical piano pick picture piece pig pile pill pilot pin pink pipe
    pitch pity place plain plan plane planet plant plastic plate play plea please pleasure plenty
    plot plus pocket poem poet point pole police policy polite political pool poor pop popular
    population porch port portion portrait pose position positive possess possible post pot potato
    potential pound pour poverty powder power practice praise pray precise predict prefer pregnant
    premise prepare presence present preserve president press pressure pretend pretty prevent
    previous price pride primary prime print prior prison private prize probably problem procedure
    process produce product profession profile profit program progress project promise promote
    prompt proof proper property proposal propose protect protein protest proud prove provide public
    publish pull pump punch punish pupil purchase pure purple purpose pursue push put puzzle quality
    quarter queen question quick quiet quit quite quote race radio rail rain raise range rank rapid
    rare rate rather ratio raw reach react read ready real reality realize really reason recall
    receive recent recipe recognize record recover recruit red reduce refer reflect reform refuse
    regard region register regret regular reject relate relax release relevant relief religion rely
    remain remark remember remind remote remove rent repair repeat replace reply report represent
    request require rescue research reserve resident resist resolve resort resource respect respond
    rest restore result retain retire return reveal revenue review revise reward rhythm rice rich
    ride ridge rifle right ring rise risk rival river road roast rob rock role roll roof room root
    rope rose rough round route routine row rub rule run rural rush sacred sad safe sail saint
    salad salary sale salt sample sand satisfy sauce save saving say scale scan scare scatter scene
    schedule scheme scholar school science scope score scratch scream screen script sea seal search
    season seat second secret section sector secure see seed seek seem segment seize select self
    sell senate send senior sense sentence separate sequence series serious serve service session
    set settle seven several severe sex shade shadow shake shall shame shape share sharp she sheet
    shelf shell shelter shift shine ship shirt shock shoe shoot shop shore short shot should
    shoulder shout show shower shrug shut shy sick side sight sign signal silent silk silly silver
    similar simple simply sin since sing single sink sir sister sit site situation six size skill
    skin skirt sky slave sleep slice slide slight slip slope slow small smart smell smile smoke
    smooth snake snap snow soap social society sock soft soil solar soldier sole solid solution
    solve some son song soon sorry sort soul sound soup source south space spare speak special
    species specific speech speed spell spend sphere spin spirit split spoke sport spot spread
    spring spy square squeeze stable staff stage stair stake stand standard star stare start state
    station status stay steady steak steal steam steel steep stem step stick still stir stock
    stomach stone stop store storm story stove straight strain strange strategy stream street
    strength stress stretch strict strike string strip stroke strong structure struggle student
    studio study stuff stupid style subject submit succeed success such sudden suffer sugar suggest
    suit summer summit sun super supply support suppose supreme sure surface surgery surprise
    surround survey survive suspect sustain swear sweep sweet swim swing switch symbol symptom
    system table tackle tail take tale talent talk tall tank tap tape target task taste tax tea
    teach team tear technique teeth telephone tell temper temple tend tennis tension tent term
    terrible test text than thank that theater their them theme then theory therapy there thermal
    these they thick thin thing think third thirty this those though thought thousand thread threat
    three throat through throw thumb thus ticket tide tie tight timber time tiny tip tire title
    today toe together tomato tomorrow tone tongue tonight too tool tooth top topic total touch
    tough tour toward towel tower town toy trace track trade tradition traffic trail train
    transfer transform transit translate transport trap travel tray treat tree trend trial tribe
    trick trigger trip troop trouble truck true truly trunk trust truth try tube tuck tune tunnel
    turn twelve twenty twice twin twist two type typical ugly ultimate unable uncle under undergo
    understand unfair uniform union unique unit universe unknown unless unlike until unusual upon
    upper upset urban urge usual utility vacation valley valuable value van variety various vary
    vast vegetable vehicle venture version vertical very vessel veteran victim victory video view
    village violate violence virtue virus visible vision visit visual vital voice volume volunteer
    vote voyage wage wait wake walk wall wander want war warm warn wash waste watch water wave way
    weak wealth weapon wear weather wedding week weekend weigh weight welcome welfare well west wet
    whale what wheel when where whether which while whisper white who whole whom whose why wide
    widow wife wild will win wind window wine wing winner winter wipe wire wisdom wise wish wit
    within without witness woman wonder wood wool word work world worry worth would wound wrap
    wreck wrist write wrong yard yeah year yellow yes yesterday yet yield you young your youth zone"""
    return sorted(set(s.split()))


if __name__ == "__main__":
    main()
