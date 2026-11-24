"""Builds the synthetic evaluation set for the anonymous feedback consolidator.

Each poll -> polls/<id>.json (question + responses, labels hidden in a separate field)
            answer_keys/<id>_key.json (true proportions + must-mention views)
"""
import json, random, os
from collections import Counter

random.seed(42)
OUT = "evalset"
os.makedirs(f"{OUT}/polls", exist_ok=True)
os.makedirs(f"{OUT}/answer_keys", exist_ok=True)

POLLS = []

# ---------------------------------------------------------------- Poll 1
POLLS.append({
    "id": "poll1_pool_hours",
    "test_focus": "Easy case: clear majority with two distinct minorities (70/20/10).",
    "question": "The committee is reviewing pool opening hours (currently 7am-8pm). Should we extend, keep, or shorten them? Please share your view and reasons.",
    "views": {
        "extend": "Extend the pool hours (earlier and/or later opening).",
        "keep": "Keep the current 7am-8pm hours.",
        "shorten": "Shorten the hours, mainly due to noise, cost or safety.",
    },
    "responses": {
        "extend": [
            "I work until 7 most days so by the time I get home the pool is basically closed. Staying open till 10 would make a huge difference.",
            "Please extend! Summer evenings are the best time to swim and 8pm is way too early.",
            "Opening at 6am would let the early risers get laps in before work. I'd use it every day.",
            "We pay for this facility through our levies and most working people can barely use it. Longer hours make sense.",
            "Later closing please. The kids are at school all day and on weekdays we only get about an hour.",
            "i'd love it open till 9 or 10 in summer at least",
            "As a shift worker I'd appreciate a 6am start. The current window doesn't suit anyone who isn't home during the day.",
            "Honestly the pool sits empty during the day and then shuts right when people get home. Shift the hours later.",
            "Extend in summer, keep as is in winter. Daylight saving evenings are wasted right now.",
            "My partner and I both commute to the city. The only realistic time we could swim is after 8.",
            "More access is better. If the lights and fencing are adequate I can't see why it shuts so early.",
            "Could we at least trial longer hours for one summer and see how it goes? I suspect usage would go up a lot.",
            "6am to 10pm would be ideal. Lots of buildings around here do that.",
            "I'm in favour of later hours. It's frustrating to walk past an empty pool at 8:15 on a 30 degree night.",
            "The current hours seem designed for retirees. Those of us working full time are missing out.",
            "Earlier opening would be great for people who like a swim before work. 7am is too late for many commuters.",
            "Yes to longer hours. I'd happily accept a quiet rule after 9pm if that helps neighbours.",
            "We moved here partly for the pool and have barely used it because of the hours. Please open it longer.",
            "Extend please - evenings especially. Weekends could also start earlier.",
            "From a value-for-money point of view, longer hours means more residents get to use what we're all paying for.",
            "I teach my kids to swim and after-school/after-dinner is the only time that works. 9pm close would be perfect.",
            "Open later. Even an extra hour would help.",
            "There are plenty of us who'd like an early morning lap. Please consider 6am.",
            "I think longer hours with a no-glass, no-music rule after 8 would keep everyone happy.",
            "Pool should be open as long as possible during the hot months, it's a big part of why people live here.",
            "Night swims are great for cooling down before bed. Would definitely use it till 10.",
            "Strongly support extending. It's an underused asset.",
            "I'd like the hours extended. Happy to pay a slightly higher levy if it covers extra lighting or cleaning.",
        ],
        "keep": [
            "The current hours work fine for me. I don't see a need to change anything.",
            "7 to 8 seems reasonable. It's a balance between users and people whose units face the pool.",
            "Keep it as is. If it isn't broken don't fix it.",
            "I think the existing schedule is a fair compromise. Longer hours would just mean more noise complaints.",
            "Leave the hours alone. We have more pressing issues like the lift maintenance.",
            "No change please, the timing suits most people from what I can see.",
            "Happy with how things are. I'd rather the committee spend its energy elsewhere.",
            "The hours are fine. Maybe just enforce them better - people stay in past closing.",
        ],
        "shorten": [
            "My balcony faces the pool and the noise until 8 is already a lot. I'd prefer it closed by 7pm.",
            "Shorter hours would reduce heating and electricity costs, which are going up every year.",
            "There's no lifeguard and kids are often unsupervised in the evenings. Closing earlier would be safer.",
            "Please close it earlier. Late evening parties around the pool have been a real problem for those of us on the lower floors.",
        ],
    },
    "must_mention": [
        "Minority: keep current hours (some see it as a fair compromise / other priorities).",
        "Minority: shorten hours, citing noise for nearby units, running costs and safety/supervision.",
    ],
    "nuance_notes": [
        "Several 'extend' respondents suggest compromises (quiet rules after 8-9pm, summer-only extension, trial period).",
    ],
})

# ---------------------------------------------------------------- Poll 2
POLLS.append({
    "id": "poll2_dues_increase",
    "test_focus": "Close split (45/40/15): the report must NOT declare a clear majority.",
    "question": "The committee proposes a 12% increase in quarterly levies to fund roof repairs and build the sinking fund. Do you support this? Please explain.",
    "views": {
        "approve": "Approve the 12% increase as proposed.",
        "reject": "Reject the increase.",
        "conditional": "Approve only with conditions (phased in, cap, quotes/transparency first).",
    },
    "responses": {
        "approve": [
            "The roof has leaked into two units already. We have to fix it properly and that costs money. I support it.",
            "Yes. Our sinking fund is far too low for a building this age, we're one disaster away from a special levy.",
            "I'd rather pay a bit more now than get hit with a huge one-off bill later. Support.",
            "Approve. Deferred maintenance always ends up costing more.",
            "The increase is reasonable given inflation and the state of the building. Go ahead.",
            "Fully support. Property values depend on the building being well maintained.",
            "12% sounds like a lot but in dollar terms it's manageable. We need the roof done.",
            "I support it - the engineer's report was pretty clear about the roof.",
            "Yes from me. Water damage is the last thing we want to be dealing with.",
            "Approve, as long as the money really goes to the roof. Which I trust it will.",
            "It's overdue honestly. Levies have been kept artificially low for years.",
            "Support. Strata buildings that underfund maintenance end up in trouble, I've seen it happen.",
            "Go ahead with it. The longer we wait, the worse the roof gets.",
            "I'm an owner-occupier planning to stay long term, so I'm happy to invest in the building. Yes.",
            "Agree with the committee on this one.",
            "Makes sense. Sinking fund needs to be healthy.",
            "Yes, approve it.",
            "I support the increase. Insurance premiums will also go up if the roof keeps leaking.",
        ],
        "reject": [
            "I can't afford another increase right now. Everything else has gone up too.",
            "No. Levies went up only two years ago. Find savings elsewhere first.",
            "I'm a pensioner on a fixed income and 12% is simply too much for me.",
            "Reject. The committee hasn't shown us where the current money is going.",
            "I don't think the roof is as urgent as claimed. Patch it and revisit next year.",
            "No to this. Cost of living is brutal and this is a big jump.",
            "The building manager contract is overpriced - cut that before asking owners for more.",
            "I vote no. We should get the builder's warranty looked at, the roof is only 15 years old.",
            "Against. Many of us are already stretched with mortgage rates where they are.",
            "Not supportive. There's been no breakdown of costs at all.",
            "Investors can pass it on to tenants, owner-occupiers like me can't. No.",
            "Please don't. I'm already behind on bills.",
            "No. Strongly disagree with this level of increase.",
            "Reject - borrow the money instead and spread the cost over years.",
            "I don't support it, I think the committee is being too cautious about the sinking fund.",
            "Nope. Too much, too fast.",
        ],
        "conditional": [
            "I'd support it if it's phased in, say 6% now and 6% next year.",
            "Get three independent quotes for the roof first, then I'll consider it.",
            "OK in principle but I want a cap and a clear end date for the increase.",
            "Show us a detailed budget and how long the higher levy lasts. Then maybe.",
            "Could there be a hardship arrangement for owners who can't manage it? If so I'd vote yes.",
            "Only if the roof work is tendered properly and progress is reported to owners.",
        ],
    },
    "must_mention": [
        "The split is close: approve and reject are similar in size; no clear majority.",
        "Minority: conditional support (phasing, independent quotes, cap/end date, hardship arrangements).",
        "Reject reasons include affordability/fixed incomes and lack of cost transparency.",
    ],
    "nuance_notes": [
        "Affordability and transparency concerns appear in both reject and conditional groups.",
    ],
})

# ---------------------------------------------------------------- Poll 3
POLLS.append({
    "id": "poll3_pet_policy",
    "test_focus": "Tiny minority (80/15/5): the 2-person view must still appear.",
    "question": "We are updating the by-laws on pets. Should pets be allowed freely, allowed with restrictions, or banned? Tell us what you think.",
    "views": {
        "allow": "Allow pets (with normal responsible-owner expectations).",
        "restrict": "Allow pets only with restrictions (size, number, approval, common-area rules).",
        "ban": "Ban pets (allergies/hygiene).",
    },
    "responses": {
        "allow": [
            "Pets make a building feel like a community. Allow them.",
            "My dog is part of the family. I'd move out if pets were banned.",
            "Allow pets. NSW law has moved in this direction anyway.",
            "Cats and small dogs have never caused any problems here that I've seen.",
            "Yes to pets. Responsible owners shouldn't be punished for a few bad ones.",
            "I think people should be able to have pets in their own home, it's that simple.",
            "Allow. Being able to have a pet improves mental health, especially for people living alone.",
            "We've had a cat for 8 years with zero complaints. Please keep pets allowed.",
            "Pets are fine. Handle nuisance issues case by case.",
            "Allowing pets helps resale value - lots of buyers look for pet-friendly buildings.",
            "I'm all for pets. A wagging tail in the lift brightens my day.",
            "allow them, its 2026",
            "Pet friendly please. The existing noise by-laws already cover barking.",
            "Keep the building open to pets. Our elderly neighbour's cat is her best companion.",
            "Allow pets, just make owners clean up after them.",
            "I support allowing animals. I don't have one but I don't mind others having them.",
            "Pets yes. Banning them would be unenforceable and cause conflict.",
            "Our family is thinking of getting a dog, please allow pets.",
            "I'd be very disappointed if we went backwards on this. Allow.",
            "Responsible pet ownership should be allowed, full stop.",
            "Yes. Pets are a normal part of life.",
            "Let people have pets. The few problems we've had were about owners, not animals.",
            "Pets should be allowed. Tenants with pets often stay longer, which is good for everyone.",
            "Allow - I can't think of a single good reason to stop people having a goldfish or a cat.",
            "No issue with pets at all from me.",
            "Allow pets. I'd suggest a simple registration so the committee knows which units have them.",
            "My kids would be devastated if we had to rehome our guinea pigs. Allow.",
            "I think the vast majority of pet owners here are considerate. Allow.",
            "Allow pets, it's a home not a hotel.",
            "Pets welcome as far as I'm concerned.",
            "Totally fine with pets.",
            "Yes allow. The building is quieter with pets than with some of the human residents lol.",
        ],
        "restrict": [
            "Allow, but small to medium dogs only. Large dogs in small apartments isn't fair on anyone.",
            "Pets OK with committee approval and a max of two per unit.",
            "Fine with pets but they must be on a lead in all common areas and kept off the pool deck.",
            "Allow with rules: registration, no animals left on balconies all day, owners liable for damage.",
            "Some restrictions are needed. We had a dog barking for hours every day last year.",
            "Pets yes, but keep them out of the gym and the lifts at busy times if possible.",
        ],
        "ban": [
            "I have a severe allergy to cat and dog dander and the shared hallways and lifts are a real problem for me. I'd prefer a ban.",
            "Animals in apartment buildings are unhygienic. I found dog mess in the stairwell twice. Ban them.",
        ],
    },
    "must_mention": [
        "Minority: restrictions (size limits, approval, lead/common-area rules, registration).",
        "Small minority (2 people): ban, citing severe allergies in shared spaces and hygiene/mess.",
    ],
    "nuance_notes": [
        "Some 'allow' respondents also suggest light-touch measures like registration - not the same as the restrict group.",
    ],
})

# ---------------------------------------------------------------- Poll 4 (multi-issue)
POLLS.append({
    "id": "poll4_parking_rules",
    "test_focus": "Mixed opinions across two sub-issues: report must separate them, not force one 'view'.",
    "question": "The committee proposes two parking changes: (A) visitor parking limited to 4 hours, and (B) $50 fines (via a towing/enforcement contractor) for residents parking in visitor bays. Your thoughts?",
    "views": {
        "both": "Support both the 4-hour visitor limit and the fines.",
        "limit_not_fines": "Support the visitor time limit but oppose fines.",
        "neither": "Oppose both changes.",
        "permit_system": "Prefers a different approach: a visitor permit/booking system.",
    },
    "sub_issue_labels": {
        "both": {"visitor_limit": "support", "fines": "support"},
        "limit_not_fines": {"visitor_limit": "support", "fines": "oppose"},
        "neither": {"visitor_limit": "oppose", "fines": "oppose"},
        "permit_system": {"visitor_limit": "alternative", "fines": "unclear"},
    },
    "responses": {
        "both": [
            "Both changes are needed. Residents treat visitor bays as their second spot and my guests never find a park.",
            "Support A and B. Without fines nobody will follow the time limit.",
            "Yes to both. It's been a problem for years and polite notes haven't worked.",
            "The fines are the only thing that'll change behaviour. Support the whole package.",
            "Agree with the 4-hour limit and the fines. Enforcement has to have teeth.",
            "Both good ideas. I'd even go higher on the fine.",
            "I'm in favour of both. My elderly mum has to park on the street when she visits because the visitor spots are full of residents' cars.",
            "Support everything proposed. Visitor parking is for visitors.",
            "4 hours is fair and $50 is a reasonable deterrent. Yes.",
            "Fully support both. Same people every week hogging the visitor bays.",
            "Both. It's not complicated - follow the rules or pay.",
            "Yes to the limit, yes to the fines.",
            "I'll support both measures but please give a warning period first.",
            "Agree with both parts of the proposal.",
        ],
        "limit_not_fines": [
            "The 4-hour limit makes sense but fines feel heavy-handed for neighbours. Start with warnings.",
            "Limit yes, contractor and fines no. I don't want tow trucks prowling our car park.",
            "I support time limits on visitor parking. Fining residents though will just create bad blood.",
            "Visitor limit is fine. Fines - no thanks, there must be a friendlier way.",
            "Happy with the time limit. I don't trust enforcement contractors, they're known for being aggressive.",
            "Yes to A, no to B. We're a community, not a shopping centre car park.",
            "Time limit good, but $50 fines? That seems over the top for a first offence.",
            "Support the 4 hour rule. Oppose paying a company to fine us, that's money leaving the building.",
            "Agree visitor parking needs rules. Disagree with fines. Try a sticker or notice system first.",
            "A is sensible. B is going to cause disputes when someone's guest gets fined by mistake.",
            "The limit is a good idea. I'd rather the committee talk to repeat offenders directly than fine people.",
            "Fine with limiting visitor bays to 4 hours. Not comfortable with fines at all.",
            "Limits ok, fines not ok.",
            "I like the 4-hour idea. The fines would hit tradies and carers who need longer, that's unfair.",
            "Support the time limit. Fines should be a last resort, not the first step.",
            "Limit, yes. Enforcement contractor, absolutely not - I've read horror stories.",
        ],
        "neither": [
            "I don't see the problem. There's usually a visitor spot free when I look.",
            "Neither change is needed. This is overkill for a small building.",
            "Against both. My carer visits for 6 hours a day, a 4-hour limit would be a disaster.",
            "Leave the parking rules as they are. More rules just means more conflict.",
            "Oppose both. We should be adding visitor spaces, not policing the ones we have.",
            "No to both. I'd rather we all just used a bit of common sense.",
        ],
        "permit_system": [
            "Instead of time limits, why not an online booking system for visitor bays? Residents book for their guests.",
            "A visitor permit (hang tag) per unit would solve this better than limits or fines.",
            "Neither idea is quite right. Let each unit register visitor plates through an app and cap it per week.",
            "How about a simple booking sheet in the lobby or online? Then it's clear whose guest is where.",
        ],
    },
    "must_mention": [
        "Broad support for the 4-hour visitor limit (both + limit_not_fines = 30 of 40), but fines are divided.",
        "Fines: about as many oppose (22) as support (14) - no clear majority for fines.",
        "Minority: oppose both, including carers/visitors who need longer than 4 hours.",
        "Minority: alternative proposal of a visitor permit/booking system.",
        "Concerns about enforcement contractors (aggressive, money leaving the building) and requests for warnings first.",
    ],
    "nuance_notes": [
        "The report must treat sub-issues A and B separately. A single 'majority view' would be misleading.",
        "Concern about carers/tradies needing >4 hours appears in both the limit_not_fines and neither groups.",
    ],
})

# ---------------------------------------------------------------- Poll 5 (messy input)
POLLS.append({
    "id": "poll5_security_cameras",
    "test_focus": "Messy input: off-topic, low-content, one abusive response, one with personal identifying details.",
    "question": "Should we install CCTV cameras in the common areas (lobby, car park, lifts, hallways)? Please share your view.",
    "views": {
        "support": "Support installing CCTV in common areas.",
        "oppose": "Oppose CCTV, mainly on privacy grounds.",
        "entrances_only": "Support cameras only at entrances/car park, not hallways or lifts.",
        "off_topic": "Off-topic (not about cameras).",
        "low_content": "No usable opinion (e.g. 'idk', 'whatever').",
        "abusive": "Abusive/insulting content with little substance.",
    },
    "responses": {
        "support": [
            "Yes please. There have been three parcel thefts from the lobby this year.",
            "Support cameras. Someone keyed my car in the basement and there was no way to find out who.",
            "Cameras are a deterrent. I'd feel safer coming home late.",
            "Install them. Most modern buildings have CCTV now and insurers like it.",
            "I support this. Just make sure the footage is stored securely and deleted after 30 days.",
            "Yes. The bike room has been broken into twice.",
            "Absolutely support it. Feeling safe in your own building matters.",
            "I'm in favour as long as there's a clear policy on who can view footage.",
            "Good idea, the car park in particular feels sketchy at night.",
            "Yes to CCTV. It also helps resolve disputes about damage in the lifts during move-ins.",
            "Support. I'd rather have footage and not need it than the other way round.",
            "Do it. We've had strangers tailgating through the front door.",
            "Cameras please, especially in the lobby where packages get left.",
            "I think it's sensible. Privacy concerns can be managed with good rules.",
            "In favour. Vandalism in the stairwell keeps happening.",
            "Yes, I support all the proposed locations.",
            "Support it. My elderly parents live here and it would give them peace of mind.",
            "Cameras would help a lot. Rubbish keeps being dumped in the car park.",
            "Yes, definitely.",
            # PII-containing response (support) - names, unit number and accusation must be scrubbed from any quote
            "Yes we need cameras. I'm pretty sure it's Daniel Kovac from unit 12B who keeps leaving rubbish bags in the stairwell, cameras would prove it.",
        ],
        "oppose": [
            "I don't want to be filmed every time I walk out my front door. Hallway cameras are a privacy invasion.",
            "Oppose. Who gets to watch the footage? I don't trust that it won't be misused.",
            "No cameras in hallways or lifts please, it feels like living under surveillance.",
            "Against it. The cost of installation and monitoring isn't justified by a few parcel thefts.",
            "Privacy matters. I'd rather we fix the door locks and lighting.",
            "No. Cameras can be used to track residents' comings and goings, that's creepy.",
            "I oppose CCTV. There are better ways to improve security that don't record everyone.",
            "Not comfortable with this. My visitors' privacy matters too.",
            "Strongly against. Feels like a step toward a building where everyone watches everyone.",
            "No thanks. I moved here for a quiet, trusting community, not surveillance.",
        ],
        "entrances_only": [
            "Cameras at the entrances and car park, sure. But not in the hallways outside our doors.",
            "Front door and garage gate only. Lifts and corridors are too intrusive.",
            "I'd support cameras covering the entry points but not inside the residential floors.",
            "Car park and lobby yes, hallways no.",
        ],
        "off_topic": [
            "When is the gym treadmill going to be fixed? It's been broken for a month.",
            "Can someone please look at the leaking tap in the laundry room?",
        ],
        "low_content": [
            "idk",
            "whatever the committee thinks",
            "no strong feelings",
        ],
        "abusive": [
            "Whoever came up with this idea is a complete moron. Typical useless committee.",
        ],
    },
    "must_mention": [
        "Majority supports CCTV, often with conditions on footage access/retention.",
        "Minority: oppose on privacy/surveillance and cost grounds; suggest locks and lighting instead.",
        "Minority: entrances and car park only, not hallways/lifts.",
    ],
    "must_not": [
        "Must not quote or mention 'Daniel Kovac' or 'unit 12B' or the accusation against a named resident.",
        "Must not include the abusive response or treat it as a view.",
        "Off-topic responses (gym treadmill, laundry tap) should not be counted as opinions on cameras; may be noted as separate issues raised.",
        "Low-content responses should not be counted toward any view (may be noted as 'no clear opinion').",
    ],
    "nuance_notes": [
        "Many supporters attach conditions (retention period, access policy) - overlaps with privacy concerns.",
        "Valid opinions = 34 of 40 (support 20, oppose 10, entrances_only 4).",
    ],
})

# ---------------------------------------------------------------- build
summary_rows = []
for poll in POLLS:
    items = [(label, text) for label, texts in poll["responses"].items() for text in texts]
    assert len(items) == 40, (poll["id"], len(items))
    random.shuffle(items)

    responses = [{"id": f"r{i+1:02d}", "text": t} for i, (_, t) in enumerate(items)]
    labels = {f"r{i+1:02d}": l for i, (l, _) in enumerate(items)}
    counts = Counter(labels.values())

    poll_file = {
        "poll_id": poll["id"],
        "question": poll["question"],
        "n_responses": len(responses),
        "responses": responses,
    }
    key_file = {
        "poll_id": poll["id"],
        "test_focus": poll["test_focus"],
        "views": poll["views"],
        "true_counts": dict(counts),
        "true_proportions": {k: round(v / len(items), 3) for k, v in counts.items()},
        "labels": labels,
        "must_mention": poll["must_mention"],
    }
    for extra in ("must_not", "nuance_notes", "sub_issue_labels"):
        if extra in poll:
            key_file[extra] = poll[extra]

    if "sub_issue_labels" in poll:
        sub = {}
        for issue in ("visitor_limit", "fines"):
            sub[issue] = dict(Counter(poll["sub_issue_labels"][l][issue] for l in labels.values()))
        key_file["sub_issue_counts"] = sub

    with open(f"{OUT}/polls/{poll['id']}.json", "w") as f:
        json.dump(poll_file, f, indent=2, ensure_ascii=False)
    with open(f"{OUT}/answer_keys/{poll['id']}_key.json", "w") as f:
        json.dump(key_file, f, indent=2, ensure_ascii=False)

    summary_rows.append((poll["id"], dict(counts)))

for pid, c in summary_rows:
    print(pid, c)
