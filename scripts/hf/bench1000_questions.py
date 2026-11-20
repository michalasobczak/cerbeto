"""Builds scripts/hf/bench1000.json: the 1000-question benchmark (bench1000.py). The 100 questions of bench.py come first,
unchanged (marked core), so old results stay comparable; the other 900 are written here from fact tables and templates,
with a fixed seed. Questions that are in the conversation data the student was tuned on (data/talk/qa_*.jsonl) are left
out, so the benchmark asks what the student knows, not what it was shown.

    python3 scripts/hf/bench1000_questions.py          (from the repo root; prints the counts per part)

An item: {"part", "kind": chat | complete | memory, "q": str or list of turns, "ok": regex, "not": regex (optional)}.
With "not" (questions with two options) the answer counts only when the right option comes before the wrong one.
"""
import ast, json, random, re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("bench1000.json")
rng = random.Random(1000)
YES, NO = r"^(yes|true)", r"^(no|false)"
NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
       "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]


def num(n: int) -> str:
    return (NUM[n] + "|" if n < len(NUM) else "") + rf"\b{n}\b"


# ---- the 100 questions of bench.py, read from its source (not imported: it runs on import) ----
src = ast.parse((REPO / "scripts/hf/bench.py").read_text())
lit = {}
for node in src.body:           # our own file: the literals evaluated with YES and NO, nothing else in scope
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("CHAT", "COMP", "MEM"):
        lit[node.targets[0].id] = eval(compile(ast.Expression(node.value), "bench.py", "eval"), {"__builtins__": {}}, {"YES": YES, "NO": NO})
items = []
for part, qs in lit["CHAT"].items():
    items += [{"part": part, "kind": "chat", "q": q, "ok": ok, "core": True} for q, ok in qs]
items += [{"part": "complete", "kind": "complete", "q": c, "ok": ok, "core": True} for c, ok in lit["COMP"]]
items += [{"part": "memory", "kind": "memory", "q": conv, "ok": ok, "core": True} for conv, ok in lit["MEM"]]
assert len(items) == 100
new = []


def chat(part, q, ok, nok=None):
    new.append({"part": part, "kind": "chat", "q": q, "ok": ok, **({"not": nok} if nok else {})})


# ---- yes/no (150 with the core 15) ----
ANIMALS = ["dog", "cat", "horse", "cow", "sheep", "pig", "lion", "tiger", "elephant", "mouse", "rabbit", "bear", "wolf",
           "fox", "monkey", "giraffe", "zebra", "goat"]
BIRDS_FLY = ["eagle", "owl", "parrot", "sparrow", "swan", "crow", "pigeon"]
FISH = ["shark", "salmon", "goldfish", "trout", "cod"]
FRUITS = ["apple", "banana", "orange", "pear", "grape", "cherry", "strawberry", "lemon", "mango", "peach", "plum"]
VEG = ["carrot", "potato", "onion", "cabbage", "lettuce", "broccoli", "pea"]
THINGS = ["car", "chair", "table", "book", "spoon", "cup", "hat", "shoe", "bus", "phone"]
a = lambda w: ("an " if w[0] in "aeiou" else "a ") + w
for w in rng.sample(ANIMALS, 10):
    chat("yes/no", f"Is {a(w)} an animal?", YES)
for w in rng.sample(THINGS, 6) + rng.sample(FRUITS, 4):
    chat("yes/no", f"Is {a(w)} an animal?", NO)
for w in rng.sample(FRUITS, 8):
    chat("yes/no", f"Is {a(w)} a fruit?", YES)
for w in rng.sample(VEG, 4) + rng.sample(ANIMALS, 3):
    chat("yes/no", f"Is {a(w)} a fruit?", NO)
for w in rng.sample(VEG, 5):
    chat("yes/no", f"Is {a(w)} a vegetable?", YES)
for w in rng.sample(BIRDS_FLY, 6):
    chat("yes/no", f"Can {a(w)} fly?", YES)
for w in ["dog", "cow", "horse", "pig", "elephant", "penguin", "fish", "snake", "rabbit", "car"]:
    chat("yes/no", f"Can {a(w)} fly?", NO)
for w in FISH:
    chat("yes/no", f"Can {a(w)} swim?", YES)
for w in FISH[:3]:
    chat("yes/no", f"Is {a(w)} a fish?", YES)
for w in ["dolphin", "cat", "frog", "duck"]:
    chat("yes/no", f"Is {a(w)} a fish?", NO)
for w in BIRDS_FLY[:3] + ["penguin", "chicken", "duck"]:
    chat("yes/no", f"Is {a(w)} a bird?", YES)
for w in ["bat", "bee", "dog", "butterfly"]:
    chat("yes/no", f"Is {a(w)} a bird?", NO)
FACTS_YES = ["Is the sea salty?", "Do fish live in water?", "Do birds have feathers?", "Do bees make honey?", "Is the grass green?",
             "Is a lemon sour?", "Is sugar sweet?", "Do plants need water?", "Does the sun rise in the morning?", "Is the Earth round?",
             "Do cows eat grass?", "Is a mountain big?", "Is fire dangerous?", "Do people need to sleep?", "Is honey sweet?",
             "Does a week have seven days?", "Is the moon in the sky at night?", "Do spiders have eight legs?", "Is a giraffe tall?",
             "Do dogs have four legs?", "Is a mouse small?", "Is milk white?", "Do trees have leaves?", "Is a snail slow?",
             "Is winter cold?", "Is a feather light?", "Can people walk?", "Is water wet?", "Do ducks swim?", "Is chocolate sweet?"]
FACTS_NO = ["Do dogs have wings?", "Is the grass blue?", "Is ice hot?", "Do cats lay eggs?", "Is a mouse bigger than a horse?",
            "Is the moon made of cheese?", "Can a pig fly?", "Is the sun cold?", "Do fish have legs?", "Is a stone soft?",
            "Do snakes have legs?", "Is a lemon sweet?", "Is the sea green and full of sugar?", "Can a car swim?", "Is a week ten days long?",
            "Is fire cold?", "Do trees walk?", "Is an elephant small?", "Is snow black?", "Can a fish climb a tree?",
            "Is a year one day long?", "Do cows say woof?", "Is a banana blue?", "Do people have three eyes?", "Is the night bright and sunny?",
            "Is a feather heavy?", "Is summer the coldest season?", "Does a spider have two legs?"]
for q in FACTS_YES:
    chat("yes/no", q, YES)
for q in FACTS_NO:
    chat("yes/no", q, NO)
TF = [("Birds can fly.", YES), ("Fish can swim.", YES), ("Snow is white.", YES), ("The sun is hot.", YES), ("Cows give milk.", YES),
      ("A cat has four legs.", YES), ("Dogs can talk.", NO), ("Ice is hot.", NO), ("A horse is smaller than an ant.", NO),
      ("Grass is purple.", NO), ("A week has seven days.", YES), ("The moon is a star.", NO)]
for s, ok in TF:
    chat("yes/no", f"{s} True or false?", ok)

# ---- preschool (150 with the core 20): opposites, colours, animal sounds, comparisons ----
OPP = [("tall", r"short"), ("long", r"short"), ("heavy", r"light"), ("hard", r"soft|easy"), ("dark", r"light|bright"),
       ("black", r"white"), ("in", r"out"), ("on", r"off"), ("left", r"right"), ("front", r"back|behind|rear"), ("yes", r"\bno\b"),
       ("good", r"bad|evil"), ("rich", r"poor"), ("loud", r"quiet|soft|silent"), ("clean", r"dirty|messy"), ("early", r"late"),
       ("near", r"far"), ("thick", r"thin"), ("strong", r"weak"), ("first", r"last"), ("high", r"low"), ("push", r"pull"),
       ("win", r"los"), ("buy", r"sell"), ("give", r"take|receive"), ("come", r"\bgo"), ("boy", r"girl"), ("man", r"woman|lady"),
       ("king", r"queen"), ("inside", r"outside"), ("above", r"below|under|beneath"), ("before", r"after"), ("start", r"end|finish|stop"),
       ("laugh", r"cry"), ("love", r"hate"), ("true", r"false|untrue"), ("cold", r"hot|warm"), ("small", r"big|large"),
       ("sad", r"happy|glad"), ("night", r"day"), ("down", r"\bup\b"), ("dry", r"wet")]
for w, ok in rng.sample(OPP, 40):
    chat("preschool", f"What is the opposite of {w}?", ok)
COL = [("a strawberry", r"red"), ("an orange", r"orange"), ("a lemon", r"yellow"), ("a carrot", r"orange"), ("coal", r"black"),
       ("milk", r"white"), ("a cherry", r"red"), ("a leaf", r"green"), ("the sun", r"yellow|orange|white"), ("chocolate", r"brown"),
       ("a cucumber", r"green"), ("blood", r"red"), ("a pumpkin", r"orange"), ("a crow", r"black"), ("a frog", r"green"),
       ("a polar bear", r"white"), ("an elephant", r"gr[ae]y"), ("the sea", r"blue"), ("broccoli", r"green"), ("a cloud", r"white|gr[ae]y"),
       ("a lime", r"green"), ("a plum", r"purple|red|violet"), ("a fire truck", r"red"), ("a flamingo", r"pink"), ("salt", r"white"),
       ("the night sky", r"black|dark|blue"), ("a pig", r"pink"), ("an apple", r"red|green|yellow"), ("gold", r"gold|yellow"),
       ("a tree trunk", r"brown"), ("sugar", r"white|brown")]
for w, ok in rng.sample(COL, 30):
    chat("preschool", f"What color is {w}?", ok)
SOUND = [("sheep", r"baa|bleat"), ("pig", r"oink|grunt"), ("horse", r"neigh|whinn"), ("lion", r"roar"), ("hen", r"cluck|bok"),
         ("rooster", r"cock-?a-?doodle|crow"), ("owl", r"hoo|twit"), ("frog", r"ribbit|croak"), ("mouse", r"squeak"), ("bee", r"buzz"),
         ("snake", r"hiss"), ("goat", r"baa|bleat|maa|meh"), ("donkey", r"hee|bray"), ("wolf", r"howl|awoo"), ("bird", r"tweet|chirp|sing"),
         ("turkey", r"gobble"), ("elephant", r"trumpet|toot"), ("bear", r"growl|roar"), ("crow", r"caw"), ("monkey", r"ooh|eek|chatter|screech")]
for w, ok in SOUND:
    chat("preschool", f"What does {a(w)} say?", ok)
SIZE = ["an ant", "a bee", "a mouse", "a cat", "a dog", "a sheep", "a horse", "an elephant", "a whale"]
pairs = [(i, j) for i in range(len(SIZE)) for j in range(len(SIZE)) if j - i >= 2]
for n, (i, j) in enumerate(rng.sample(pairs, 26)):
    small, big = SIZE[i], SIZE[j]
    word = "bigger" if n % 2 == 0 else "smaller"
    x, y = (small, big) if rng.random() < 0.5 else (big, small)
    right, wrong = (big, small) if word == "bigger" else (small, big)
    if f"Which is {word}, {x} or {y}?" == "Which is bigger, an elephant or a mouse?":
        continue
    chat("preschool", f"Which is {word}, {x} or {y}?", right.split()[-1], wrong.split()[-1])
CMP = [("hotter", "the sun", "the moon"), ("colder", "ice", "fire"), ("heavier", "a rock", "a feather"),
       ("lighter", "a feather", "a brick"), ("faster", "a car", "a bike"), ("slower", "a snail", "a horse"),
       ("taller", "a giraffe", "a dog"), ("shorter", "a mouse", "a tree"), ("faster", "a cheetah", "a turtle"),
       ("heavier", "an elephant", "a cat"), ("louder", "a lion", "a mouse"), ("sweeter", "honey", "a lemon"),
       ("longer", "a snake", "a worm"), ("higher", "a mountain", "a hill"), ("older", "a grandfather", "a baby"),
       ("bigger", "the sun", "the earth"), ("deeper", "the ocean", "a pond"), ("wetter", "the sea", "the desert")]
for word, right, wrong in CMP:
    x, y = (right, wrong) if rng.random() < 0.5 else (wrong, right)
    chat("preschool", f"Which is {word}, {x} or {y}?", right.split()[-1], wrong.split()[-1])

# ---- knowledge (250 with the core 30) ----
CAP = [("Germany", "berlin"), ("Spain", "madrid"), ("Russia", "moscow"), ("China", "beijing|peking"), ("the United States", "washington"),
       ("Canada", "ottawa"), ("Egypt", "cairo"), ("Greece", "athens"), ("Poland", "warsaw"), ("Ireland", "dublin"),
       ("Scotland", "edinburgh"), ("Portugal", "lisbon"), ("Austria", "vienna"), ("Norway", "oslo"), ("Sweden", "stockholm"),
       ("Denmark", "copenhagen"), ("Finland", "helsinki"), ("the Netherlands", "amsterdam"), ("Belgium", "brussels"),
       ("Hungary", "budapest"), ("the Czech Republic", "prague"), ("India", "delhi"), ("Mexico", "mexico city"),
       ("Argentina", "buenos aires"), ("Kenya", "nairobi"), ("Thailand", "bangkok"), ("South Korea", "seoul"), ("Cuba", "havana"),
       ("Peru", "lima"), ("Australia", "canberra"), ("Turkey", "ankara"), ("Brazil", "brasilia|brasília")]
for c, ok in CAP:
    chat("knowledge", f"What is the capital of {c}?", ok)
LEGS = [("a cat", 4), ("a horse", 4), ("a cow", 4), ("an insect", 6), ("an ant", 6), ("a bee", 6), ("a chicken", 2), ("a duck", 2),
        ("a person", 2), ("a beetle", 6)]
for w, n in LEGS:
    chat("knowledge", f"How many legs does {w} have?", num(n))
chat("knowledge", "How many legs does a snake have?", r"zero|none|\bno\b|\b0\b|not have|doesn't have|does not have")
chat("knowledge", "How many arms does an octopus have?", num(8))
chat("knowledge", "How many arms does a starfish usually have?", num(5))
BABY = [("sheep", r"lamb"), ("horse", r"foal|colt|filly"), ("pig", r"piglet"), ("duck", r"duckling"), ("goat", r"\bkid"),
        ("frog", r"tadpole|froglet"), ("bear", r"cub"), ("lion", r"cub"), ("chicken", r"chick"), ("kangaroo", r"joey"),
        ("deer", r"fawn"), ("swan", r"cygnet"), ("goose", r"gosling"), ("owl", r"owlet"), ("butterfly", r"caterpillar|larva")]
for w, ok in BABY:
    chat("knowledge", f"What is a baby {w} called?", ok)
HOME = [("a bird", r"nest|tree"), ("a bee", r"hive|nest"), ("a spider", r"web"), ("a rabbit", r"burrow|hole|warren|hutch"),
        ("a horse", r"stable|barn|field|farm"), ("a lion", r"den|savann|jungle|africa|grass|pride"), ("a bear", r"cave|den|forest|woods"),
        ("an ant", r"anthill|ant hill|nest|colony|ground"), ("a pig", r"sty|pen|farm|mud"), ("a dog", r"kennel|house|home")]
for w, ok in HOME:
    chat("knowledge", f"Where does {w} live?", ok)
FOOD = [("cows", r"grass|hay"), ("rabbits", r"carrot|grass|vegetable|lettuce|hay|plant"), ("monkeys", r"banana|fruit"),
        ("pandas", r"bamboo"), ("cats", r"mice|mouse|fish|meat|cat food"), ("bees", r"nectar|pollen|flower|honey"),
        ("koalas", r"eucalyptus|leaves"), ("sheep", r"grass|hay"), ("birds", r"seed|worm|insect|bug|berr"), ("horses", r"grass|hay|oat|carrot|apple")]
for w, ok in FOOD:
    chat("knowledge", f"What do {w} eat?", ok)
GIVE = [("chickens", r"egg"), ("sheep", r"wool|milk|meat"), ("bees", r"honey|wax")]
for w, ok in GIVE:
    chat("knowledge", f"What do {w} give us?", ok)
BODY = [("What do we use to smell?", r"nose"), ("What do we use to taste?", r"tongue|mouth"), ("What do we use to think?", r"brain|head|mind"),
        ("What do we breathe with?", r"lung|nose|mouth"), ("How many fingers are on one hand?", num(5)),
        ("How many toes are on one foot?", num(5)), ("How many eyes do people have?", num(2)), ("What do we use our teeth for?", r"chew|bit|eat"),
        ("What does the stomach do?", r"digest|food"), ("What do we use to walk?", r"leg|feet|foot"), ("What covers our body?", r"skin|cloth"),
        ("What do bones make?", r"skeleton|body|blood|frame")]
for q, ok in BODY:
    chat("knowledge", q, ok)
TIME = [("How many hours are in a day?", num(24) + "|twenty-four|twenty four"), ("How many minutes are in an hour?", r"sixty|\b60\b"),
        ("How many seconds are in a minute?", r"sixty|\b60\b"), ("How many days are in a year?", r"\b365\b|three hundred"),
        ("How many seasons are there?", num(4)), ("What season comes after spring?", r"summer"),
        ("What season comes after summer?", r"autumn|fall"), ("What season comes after autumn?", r"winter"),
        ("What day comes after Monday?", r"tuesday"), ("What day comes before Friday?", r"thursday"),
        ("What day comes after Saturday?", r"sunday"), ("What is the first month of the year?", r"january"),
        ("What is the last month of the year?", r"december"), ("What month comes after March?", r"april"),
        ("What month comes before June?", r"may"), ("How many days are in a weekend?", num(2)),
        ("In which season do leaves fall from the trees?", r"autumn|fall"), ("In which season does it snow?", r"winter"),
        ("What is the hottest season?", r"summer"), ("What do we call the time when the sun goes down?", r"evening|sunset|dusk|night")]
for q, ok in TIME:
    chat("knowledge", q, ok)
SCI = [("Which planet is called the red planet?", r"mars"), ("Which planet has big rings?", r"saturn"),
       ("What does the Moon go around?", r"earth"), ("What do we call frozen water?", r"\bice"), ("What is steam made of?", r"water"),
       ("At what temperature does water freeze in Celsius?", r"zero|\b0\b"), ("At what temperature does water boil in Celsius?", r"hundred|\b100\b"),
       ("What falls from the clouds?", r"rain|snow|water|hail"), ("What color are most leaves?", r"green"),
       ("What force pulls things down to the ground?", r"gravity"), ("What do magnets pull?", r"iron|metal|steel"),
       ("What is H2O?", r"water"), ("What do plants make their food from?", r"sun|light|water|air|carbon"),
       ("What do we call the gas that plants take in?", r"carbon"), ("What is the Sun?", r"star"),
       ("What do we call a scientist who studies stars?", r"astronom"), ("What do bees carry from flower to flower?", r"pollen|nectar"),
       ("What is the hardest natural thing?", r"diamond"), ("What do we call animals that eat only plants?", r"herbivore|vegetarian"),
       ("What do we call animals that eat meat?", r"carnivore|predator"), ("What is the opposite of north?", r"south"),
       ("Where does the sun rise?", r"east"), ("Where does the sun set?", r"west"), ("What makes a shadow?", r"light|sun|block"),
       ("What do we call a very big wave?", r"tsunami|tidal")]
for q, ok in SCI:
    chat("knowledge", q, ok)
ANIM = [("Which animal has a long neck?", r"giraffe"), ("Which animal has a trunk?", r"elephant"),
        ("Which animal has black and white stripes?", r"zebra|skunk|tiger|badger"), ("Which bird cannot fly and lives where it is cold?", r"penguin"),
        ("What is the fastest land animal?", r"cheetah"), ("What is the largest land animal?", r"elephant"), ("What is the tallest animal?", r"giraffe"),
        ("Which animal carries its baby in a pouch?", r"kangaroo|koala|opossum|wallaby"), ("Which animal gives us wool?", r"sheep|lamb|alpaca|llama"),
        ("Which insect makes honey?", r"\bbee"), ("Which animal has a shell and moves very slowly?", r"tortoise|turtle|snail"),
        ("Which animal can change its color?", r"chameleon|octopus|squid|cuttle"), ("Which animal hops and has long ears?", r"rabbit|bunny|hare|kangaroo"),
        ("Which animal has a mane?", r"lion|horse"), ("Which animal is known for being very slow?", r"sloth|snail|tortoise|turtle"),
        ("Which bird can say words?", r"parrot|mynah|myna"), ("Which animal sleeps all winter?", r"bear|hedgehog|dormouse|bat|tortoise"),
        ("What animal is a puppy?", r"dog"), ("Which animal has eight legs?", r"spider|octopus|scorpion|tick"),
        ("Which sea animal has eight arms?", r"octopus")]
for q, ok in ANIM:
    chat("knowledge", q, ok)
GEO = [("What is the largest ocean?", r"pacific"), ("On which continent is Egypt?", r"africa"), ("In which city is the Eiffel Tower?", r"paris"),
       ("How many continents are there?", num(7)), ("What is the tallest mountain in the world?", r"everest"),
       ("What is the coldest continent?", r"antarctic"), ("Which country is famous for kangaroos?", r"australia"),
       ("In which country are the pyramids of Giza?", r"egypt"), ("Which country is shaped like a boot?", r"italy"),
       ("In which city is Big Ben?", r"london"), ("In which city is the Statue of Liberty?", r"new york"),
       ("What is the longest river in the world?", r"nile|amazon"), ("What is a large area of sand with very little rain?", r"desert"),
       ("What do we call a piece of land with water all around it?", r"island"), ("What is the biggest desert in the world?", r"sahara|antarctic"),
       ("On which continent is Brazil?", r"south america|america"), ("On which continent is China?", r"asia"),
       ("On which continent is France?", r"europe"), ("What ocean is between Europe and America?", r"atlantic"),
       ("What language do people speak in Spain?", r"spanish"), ("What language do people speak in France?", r"french"),
       ("What language do people speak in Germany?", r"german"), ("What is the biggest country in the world?", r"russia")]
for q, ok in GEO:
    chat("knowledge", q, ok)
PEOPLE = [("Who painted the Mona Lisa?", r"leonardo|da vinci|vinci"), ("Who was the first person to walk on the Moon?", r"armstrong"),
          ("Who discovered gravity?", r"newton"), ("Who invented the light bulb?", r"edison"),
          ("Who was the first president of the United States?", r"washington"), ("Who wrote the Harry Potter books?", r"rowling"),
          ("Who came up with the theory of relativity?", r"einstein"), ("Who sailed to America in 1492?", r"columbus"),
          ("Who discovered penicillin?", r"fleming"), ("Who wrote Hamlet?", r"shakespeare")]
for q, ok in PEOPLE:
    chat("knowledge", q, ok)
USE = [("What do we use to cut paper?", r"scissor|knife"), ("What do we write with?", r"pen|pencil|crayon|chalk"),
       ("What do we sleep in?", r"\bbed"), ("What do we use to tell the time?", r"clock|watch"), ("What do we wear on our feet?", r"shoe|sock|boot|sandal|slipper"),
       ("What do we use an umbrella for?", r"rain|wet|sun"), ("What do we use to brush our teeth?", r"brush"),
       ("What do we eat soup with?", r"spoon"), ("Where do we keep food cold?", r"fridge|refrigerator|freezer"),
       ("What does a doctor do?", r"help|sick|heal|treat|care|ill|medicine|patient"), ("What does a teacher do?", r"teach|school|learn|help"),
       ("What does a farmer do?", r"grow|farm|crop|animal|food|plant"), ("What does a firefighter do?", r"fire"),
       ("What does a pilot fly?", r"plane|airplane|aeroplane|jet|aircraft|helicopter"), ("What does a baker make?", r"bread|cake|bun|pastr|cookie|pie"),
       ("What does a dentist look after?", r"teeth|tooth"), ("What do we use to open a locked door?", r"key"),
       ("What do we use to dry ourselves after a bath?", r"towel"), ("What do we put on our head when it is cold?", r"hat|cap|hood|beanie"),
       ("What do we use to see in the dark?", r"torch|flashlight|light|lamp|candle")]
for q, ok in USE:
    chat("knowledge", q, ok)
KIND = [("What is a rose?", r"flower|plant"), ("What kind of animal is a whale?", r"mammal"), ("What kind of animal is a frog?", r"amphibian"),
        ("What is a robin?", r"bird"), ("Is an apple a fruit or a vegetable?", r"fruit", r"vegetable"),
        ("Is a potato a fruit or a vegetable?", r"vegetable", r"fruit"), ("Is a shark a fish or a mammal?", r"fish", r"mammal"),
        ("Is a dolphin a fish or a mammal?", r"mammal", r"fish"), ("Is a bat a bird or a mammal?", r"mammal", r"bird"),
        ("Is a spider an insect or an arachnid?", r"arachnid", r"insect"), ("Is the Sun a star or a planet?", r"star", r"planet"),
        ("Is the Earth a star or a planet?", r"planet", r"star")]
for q, ok, *nok in KIND:
    chat("knowledge", q, ok, nok[0] if nok else None)
SHAPE = [("How many sides does a square have?", num(4)), ("How many sides does a hexagon have?", num(6)),
         ("How many sides does a pentagon have?", num(5)), ("How many sides does an octagon have?", num(8)),
         ("How many corners does a triangle have?", num(3)), ("What shape is a ball?", r"round|sphere|circle"),
         ("What shape is a wheel?", r"round|circle"), ("How many wheels does a bicycle have?", num(2)),
         ("How many wheels does a car have?", num(4)), ("How many wheels does a tricycle have?", num(3)),
         ("How many players are in a football team?", r"eleven|\b11\b"), ("How many sides does a circle have?", r"zero|none|\bno\b|one|\b0\b|\b1\b")]
for q, ok in SHAPE:
    chat("knowledge", q, ok)

# ---- arithmetic (100 with the core 10) ----
seen = {q for q, _ in lit["CHAT"]["arithmetic"]}
w = lambda n: NUM[n] if rng.random() < 0.7 else str(n)
while sum(1 for x in new if x["part"] == "arithmetic") < 90:
    op = rng.choice(["plus", "plus", "minus", "minus", "times"])
    if op == "plus":
        x, y = rng.randint(1, 10), rng.randint(1, 10); r = x + y
    elif op == "minus":
        x = rng.randint(2, 20); y = rng.randint(1, x); r = x - y
    else:
        x, y = rng.randint(1, 5), rng.randint(1, 4); r = x * y
    form = rng.choice(["What is {x} {op} {y}?", "How much is {x} {op} {y}?", "{x} {op} {y} is what?"])
    q = form.format(x=w(x), op=op, y=w(y))
    q = q[0].upper() + q[1:]
    if q in seen:
        continue
    seen.add(q)
    chat("arithmetic", q, num(r))


# ---- more of the same kinds, to reach 1000 after the leak check ----
for q in ["Is a tiger a big cat?", "Do frogs live near water?", "Is the sky blue on a sunny day?", "Do apples grow on trees?",
          "Is a whale bigger than a fish?", "Do bees fly?", "Is ice frozen water?", "Do babies grow up?", "Is the sun a star?",
          "Can a kangaroo jump?", "Do owls fly at night?", "Is a carrot orange?", "Do horses eat grass?", "Is the ocean deep?",
          "Can a monkey climb trees?", "Is bread made from flour?"]:
    chat("yes/no", q, YES)
for q in ["Is a carrot blue?", "Can a rock swim?", "Do cows fly?", "Is the moon hot like the sun?", "Do birds have four legs?",
          "Is a tree an animal?", "Can a baby drive a car?", "Is salt sweet?", "Do fish breathe air like people?",
          "Is a cat bigger than an elephant?", "Do penguins live in the desert?", "Is January in summer in England?"]:
    chat("yes/no", q, NO)
for w, ok in [("hot", r"cold|cool"), ("old", r"new|young"), ("happy", r"sad|unhappy"), ("open", r"clos|shut"), ("full", r"empty"),
              ("light", r"dark|heavy"), ("quiet", r"loud|noisy"), ("easy", r"hard|difficult"), ("happy", r"sad"), ("young", r"old"),
              ("asleep", r"awake"), ("question", r"answer"), ("friend", r"enemy|foe"), ("sweet", r"sour|bitter|salty"),
              ("wide", r"narrow"), ("fat", r"thin|skinny|slim"), ("summer", r"winter"), ("north", r"south"), ("east", r"west")]:
    chat("preschool", f"What is the opposite of {w}?", ok)
for w, ok in [("a raspberry", r"red|pink"), ("a swan", r"white|black"), ("a penguin", r"black|white"), ("a lobster", r"red|blue|orange"),
              ("an egg yolk", r"yellow|orange"), ("a mouse", r"gr[ae]y|brown|white"), ("a tiger", r"orange|black|stripe"),
              ("a sunflower", r"yellow"), ("a ladybird", r"red|black|spot")]:
    chat("preschool", f"What color is {w}?", ok)
for word, right, wrong in [("bigger", "a bus", "a ball"), ("smaller", "a pea", "a pumpkin"), ("heavier", "a car", "an apple"),
                           ("hotter", "an oven", "a fridge"), ("faster", "a plane", "a boat"), ("taller", "a house", "a chair")]:
    x, y = (right, wrong) if rng.random() < 0.5 else (wrong, right)
    chat("preschool", f"Which is {word}, {x} or {y}?", right.split()[-1], wrong.split()[-1])
for q, ok in [("What is the capital of Switzerland?", r"bern"), ("What is the capital of Ukraine?", r"kyiv|kiev"),
              ("What is the capital of Romania?", r"bucharest"), ("What is the capital of Croatia?", r"zagreb"),
              ("What is the capital of Iceland?", r"reykjav"), ("What is the capital of New Zealand?", r"wellington"),
              ("What is the capital of Nigeria?", r"abuja"), ("What is the capital of Colombia?", r"bogot"),
              ("What is the capital of Vietnam?", r"hanoi"), ("What is the capital of Indonesia?", r"jakarta"),
              ("What is the capital of the Philippines?", r"manila"), ("What is the capital of Wales?", r"cardiff"),
              ("What is the capital of Bulgaria?", r"sofia"), ("What is the capital of Serbia?", r"belgrade"),
              ("Which planet is closest to the Sun?", r"mercury"), ("How many planets are in our solar system?", num(8)),
              ("What do we call a group of wolves?", r"pack"), ("What do we call a group of fish?", r"school|shoal"),
              ("What do we call a group of cows?", r"herd"), ("What do we call a group of birds?", r"flock"),
              ("What is the largest bird?", r"ostrich"), ("What is the smallest bird?", r"hummingbird|bee hummingbird"),
              ("Which animal is the biggest cat?", r"tiger|lion"), ("What do frogs eat?", r"insect|fl(y|ies)|bug|worm"),
              ("What do spiders catch in their webs?", r"insect|fl(y|ies)|bug"), ("Which animal is a symbol of China?", r"panda|dragon"),
              ("What is the main gas in the air?", r"nitrogen"), ("What do plants give off that we breathe?", r"oxygen"),
              ("What is the boiling point of water in Fahrenheit?", r"\b212\b"), ("How many bones are in the human body?", r"\b206\b"),
              ("What organ pumps blood?", r"heart"), ("What organ do we think with?", r"brain"), ("What do we call baby teeth falling out?", r"tooth|teeth|loose|lose|fall"),
              ("What do we need to see?", r"eye|light"), ("Which month has the fewest days?", r"february"),
              ("How many days are in February in a normal year?", r"twenty-eight|twenty eight|\b28\b"),
              ("How many days are in a leap year?", r"\b366\b"), ("How many weeks are in a year?", r"fifty-two|fifty two|\b52\b"),
              ("What is a baby goat called?", r"\bkid"), ("Which animal is called man's best friend?", r"dog"),
              ("What do caterpillars eat?", r"leaves|leaf|plant"), ("Where do polar bears live?", r"arctic|north|ice|snow|cold"),
              ("Where do camels live?", r"desert"), ("What does a camel have on its back?", r"hump"),
              ("What do we call the home of a beaver?", r"lodge|dam"), ("What is a female sheep called?", r"\bewe"),
              ("What is a male chicken called?", r"rooster|cock"), ("What is a female horse called?", r"mare"),
              ("What metal is yellow and very valuable?", r"gold"), ("What do we call money made of paper?", r"note|bill|cash|banknote"),
              ("Which instrument has black and white keys?", r"piano|keyboard|organ"), ("How many strings does a normal guitar have?", num(6)),
              ("What is the color of a stop light?", r"red"), ("What color means go at traffic lights?", r"green"),
              ("Which sport uses a bat and a ball and wickets?", r"cricket"), ("In which sport do you score a touchdown?", r"football|rugby")]:
    chat("knowledge", q, ok)
for c, ok in [("A group of wolves is called a", r"pack"), ("A group of cows is called a", r"herd"), ("The closest planet to the Sun is", r"mercury"),
              ("A camel lives in the", r"desert"), ("Paper is made from", r"wood|tree|pulp"), ("Glass is made from", r"sand"),
              ("Bread is made from", r"flour|wheat|grain|dough"), ("Cheese is made from", r"milk")]:
    new.append({"part": "complete", "kind": "complete", "q": c, "ok": ok})

# ---- completing (150 with the core 20) ----
COMPL = [("The sun rises in the", r"east"), ("The sun sets in the", r"west"), ("A week has seven", r"day"), ("A year has twelve", r"month"),
         ("Two plus two is", r"four|\b4\b"), ("The opposite of up is", r"down"), ("A spider has eight", r"leg"), ("Fish live in", r"water|sea|river|ocean|lake|pond|tank"),
         ("Birds build", r"nest"), ("The moon shines at", r"night"), ("We sleep in a", r"bed"), ("We eat soup with a", r"spoon"),
         ("We cut paper with", r"scissor"), ("Apples grow on", r"tree"), ("A baby dog is called a", r"pup"), ("A baby sheep is called a", r"lamb"),
         ("Grass is", r"green"), ("Bananas are", r"yellow"), ("In winter it is", r"cold|snow"), ("In summer it is", r"hot|warm|sun"),
         ("After Monday comes", r"tuesday"), ("After March comes", r"april"), ("Seven, eight, nine,", r"ten|\b10\b"), ("A, B, C,", r"\bd\b"),
         ("Red, orange, yellow, green,", r"blue"), ("Spring, summer, autumn,", r"winter"), ("Ten, twenty, thirty,", r"forty|\b40\b"),
         ("Twinkle, twinkle, little", r"star"), ("Humpty Dumpty sat on a", r"wall"), ("Jack and Jill went up the", r"hill"),
         ("Mary had a little", r"lamb"), ("Once upon a", r"time"), ("Happy birthday to", r"you"), ("Thank you very", r"much"),
         ("Ladies and", r"gentlem"), ("Salt and", r"pepper"), ("Bread and", r"butter"), ("Fish and", r"chip"), ("Day and", r"night"),
         ("Black and", r"white"), ("Knife and", r"fork"), ("Cats and", r"dog"), ("Ready, steady,", r"\bgo"), ("The early bird catches the", r"worm"),
         ("An apple a day keeps the", r"doctor"), ("The Eiffel Tower is in", r"paris|france"), ("The pyramids are in", r"egypt"),
         ("Kangaroos live in", r"australia"), ("Polar bears live in the", r"arctic|north|cold|ice|snow"), ("The tallest animal is the", r"giraffe"),
         ("The fastest land animal is the", r"cheetah"), ("The king of the jungle is the", r"lion"), ("Honey is made by", r"\bbee"),
         ("Milk comes from", r"cow|goat"), ("Eggs come from", r"hen|chicken|bird"), ("Wool comes from", r"sheep"),
         ("The largest ocean is the", r"pacific"), ("We breathe with our", r"lung|nose|mouth"), ("We see with our", r"eye"),
         ("We hear with our", r"\bear"), ("We smell with our", r"nose"), ("We walk with our", r"leg|feet|foot"), ("A car has four", r"wheel|door|tyre|tire"),
         ("A bicycle has two", r"wheel|tyre|tire|pedal"), ("A square has four", r"side|corner|angle|edge"), ("There are seven days in a", r"week"),
         ("There are sixty minutes in an", r"hour"), ("The first month of the year is", r"january"), ("The last month of the year is", r"december"),
         ("Ice melts when it gets", r"warm|hot"), ("Plants grow from", r"seed|soil|ground"), ("Caterpillars turn into", r"butterfl|moth"),
         ("Tadpoles turn into", r"frog|toad"), ("At night we can see the", r"moon|star"), ("The opposite of big is", r"small|little"),
         ("The opposite of happy is", r"sad"), ("The opposite of day is", r"night"), ("The opposite of wet is", r"dry"), ("The opposite of fast is", r"slow"),
         ("A doctor helps people who are", r"sick|ill|hurt|injur"), ("A teacher works in a", r"school|class"), ("A farmer works on a", r"farm|field"),
         ("A pilot flies a", r"plane|jet|aircraft|helicopter"), ("A baker makes", r"bread|cake|bun|pastr"), ("Firefighters put out", r"fire|flame"),
         ("Gravity was discovered by", r"newton"), ("The Mona Lisa was painted by", r"leonardo|vinci"), ("The first man on the Moon was", r"armstrong|neil"),
         ("Water is made of hydrogen and", r"oxygen"), ("Most of the Earth is covered by", r"water|ocean|sea"), ("The Sun is a", r"star"),
         ("The Moon goes around the Earth, and the Earth goes around the", r"\bsun\b"), ("The capital of England is", r"london"),
         ("The capital of Italy is", r"rome"), ("The capital of Germany is", r"berlin"), ("The capital of Spain is", r"madrid"),
         ("The capital of Japan is", r"tokyo"), ("The capital of Russia is", r"moscow"), ("The red planet is", r"mars"),
         ("The planet with big rings is", r"saturn"), ("We live on the planet", r"earth"), ("Cows say", r"moo"), ("Dogs say", r"woof|bark|ruff"),
         ("Ducks say", r"quack"), ("Cats say", r"meow|miaow"), ("Sheep say", r"baa"), ("A baby cow is called a", r"calf"),
         ("A baby horse is called a", r"foal"), ("A baby frog is called a", r"tadpole"), ("Bees live in a", r"hive|nest"),
         ("Spiders make", r"web|silk"), ("A triangle has three", r"side|corner|angle|point"), ("Five plus five is", r"ten|\b10\b"),
         ("Ten minus one is", r"nine|\b9\b"), ("Three plus three is", r"six|\b6\b"), ("One, two, three,", r"four|\b4\b"),
         ("Monday, Tuesday, Wednesday,", r"thursday"), ("April, May,", r"june"), ("Winter, spring,", r"summer"),
         ("Rain falls from the", r"cloud|sky"), ("Snow is cold and", r"white|wet|soft"), ("Lemons taste", r"sour"), ("Sugar tastes", r"sweet"),
         ("Fire is", r"hot|red|danger|burn|bright|orange"), ("Ice is", r"cold|frozen|water|slippery|hard"), ("A giraffe has a long", r"neck"),
         ("An elephant has a long", r"trunk|nose"), ("A rabbit has long", r"ear"), ("A zebra has black and white", r"stripe"),
         ("Penguins cannot", r"fly"), ("Fish breathe with", r"gill"), ("Birds have two", r"wing|leg|feet|eye"),
         ("The heart is a", r"muscle|organ|pump"), ("Shakespeare wrote", r"romeo|hamlet|play|macbeth|sonnet|poem"),
         ("The longest river in the world is the", r"nile|amazon"), ("The tallest mountain in the world is", r"everest"),
         ("Water boils at one hundred", r"degree"), ("The Statue of Liberty is in", r"new york|america|united states|usa"),
         ("Big Ben is in", r"london|england|britain|uk"), ("Christmas is in", r"december|winter")]
done = {c for c, _ in lit["COMP"]}
for c, ok in COMPL:
    if c not in done:
        new.append({"part": "complete", "kind": "complete", "q": c, "ok": ok})

# ---- memory (100 with the core 5): facts in one or two turns, then a question about one of them ----
NAMES = ["Anna", "Ben", "Clara", "David", "Ella", "Felix", "Grace", "Harry", "Ivy", "Jonas", "Kate", "Liam", "Lucy", "Max", "Nina",
         "Oscar", "Paul", "Rosa", "Sam", "Tina", "Victor", "Wendy", "Yusuf", "Zara"]
PETS = ["cat", "dog", "rabbit", "horse", "parrot", "hamster", "fish", "turtle"]
PETN = ["Bella", "Charlie", "Daisy", "Milo", "Luna", "Oliver", "Coco", "Rocky", "Pepper", "Biscuit", "Ginger", "Shadow", "Bubbles", "Toby"]
COLS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "black"]
FOODS = ["pizza", "pasta", "apples", "soup", "pancakes", "rice", "bananas", "cheese"]
CITIES = ["London", "Paris", "Berlin", "Madrid", "Rome", "Vienna", "Dublin", "Oslo"]
SPORTS = ["football", "tennis", "swimming", "basketball", "running", "skiing"]


def fact(kind):
    if kind == "pet":
        p, n = rng.choice(PETS), rng.choice(PETN)
        return f"I have {a(p)} called {n}.", f"What is my {p} called?", n.lower()
    if kind == "color":
        c = rng.choice(COLS); return f"My favourite color is {c}.", "What is my favourite color?", c
    if kind == "food":
        f = rng.choice(FOODS); return f"My favourite food is {f}.", "What is my favourite food?", f.rstrip("s")
    if kind == "city":
        c = rng.choice(CITIES); return f"I live in {c}.", "Where do I live?", c.lower()
    if kind == "sport":
        s = rng.choice(SPORTS); return f"I like to play {s}." if s in ("football", "tennis", "basketball") else f"I like {s}.", "What sport do I like?", {"swimming": "swim", "running": "run", "skiing": "ski"}.get(s, s)
    if kind == "age":
        n = rng.randint(5, 12); return f"I am {NUM[n]} years old.", "How old am I?", num(n)
    if kind == "sister":
        n = rng.choice(NAMES); return f"My sister is called {n}.", "What is my sister's name?", n.lower()


KINDS = ["pet", "color", "food", "city", "sport", "age", "sister"]
mem = set()
while sum(1 for x in new if x["part"] == "memory") < 135:
    me = rng.choice(NAMES)
    f1 = fact(rng.choice(KINDS))
    two = rng.random() < 0.5
    turns = [f"Hi! My name is {me}.", f1[0]]
    ask, ok = f1[1], f1[2]
    if two:
        k2 = rng.choice([k for k in KINDS if f1[1] != fact(k)[1]])
        f2 = fact(k2)
        turns.append(f2[0])
        if rng.random() < 0.5:
            ask, ok = f2[1], f2[2]
    if rng.random() < 0.15:
        ask, ok = "What is my name?", me.lower()
    conv = (*turns, ask)
    if conv in mem:
        continue
    mem.add(conv)
    new.append({"part": "memory", "kind": "memory", "q": [*turns, ask], "ok": ok})

# synonyms the first run on v6 showed (2026-11-21): "grandpa" for grandfather, "soccer" for football
SYN = {"grandfather": "grandfather|grandpa|grandad|granddad", "football": "football|soccer"}
for x in new:
    x["ok"] = SYN.get(x["ok"], x["ok"])

# ---- leave out what the student was tuned on, and duplicates ----
norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
seen_q = set()
for f in (REPO / "data/talk").glob("qa_*.jsonl"):
    seen_q |= {norm(json.loads(x)["q"]) for x in f.read_text().splitlines() if x.strip()}
core_q = {norm(x["q"] if isinstance(x["q"], str) else " / ".join(x["q"])) for x in items}
kept, leaked, dup = [], [], 0
for x in new:
    k = norm(x["q"] if isinstance(x["q"], str) else " / ".join(x["q"]))
    if k in core_q or any(k == norm(y["q"] if isinstance(y["q"], str) else " / ".join(y["q"])) for y in kept):
        dup += 1
    elif x["kind"] == "chat" and k in seen_q:
        leaked.append(x["q"])
    else:
        kept.append(x)
core_leaks = [x["q"] for x in items if x["kind"] == "chat" and norm(x["q"]) in seen_q]
for x in kept:
    re.compile(x["ok"]); x.get("not") and re.compile(x["not"])
target = {"yes/no": 150, "preschool": 140, "knowledge": 250, "arithmetic": 90, "complete": 140, "memory": 130}
by = {p: [x for x in kept if x["part"] == p] for p in target}
final = list(items)
for p, n in target.items():
    assert len(by[p]) >= n, (p, len(by[p]), n)
    final += by[p][:n]
    print(f"{p:11} written {len(by[p]):4}  used {min(n, len(by[p])):4}  (+ core {sum(1 for x in items if x['part'] == p)})")
print(f"left out: {len(leaked)} in the tuning data, {dup} duplicates; core questions also in the tuning data: {len(core_leaks)}")
print("total", len(final)); assert len(final) == 1000
OUT.write_text(json.dumps(final, indent=0, ensure_ascii=False))
