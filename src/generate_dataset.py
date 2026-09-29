#!/usr/bin/env python3
"""
Generate a synthetic localized-learning benchmark for lexical binding, factual association, behavioral policy learning, causal mapping, and procedural reasoning.

Outputs:
  data/latent_specs.jsonl
  data/prompt_examples.jsonl
  data/prompt_examples_flat.csv

This generator is deterministic by seed and intentionally separates:
  1. latent specifications: what the model must learn
  2. prompt examples: train/eval/generalization/control instances generated from each spec

Later calibration should operate over latent specs, not isolated prompt rows.

Important design choice:
  Training prompts are retrieval/cloze style. The target answer does NOT appear
  inside the prompt. It appears only in the supervised completion/target field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import csv
import hashlib
import json
import random
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


# ---------------------------------------------------------------------
# Source vocabularies
# ---------------------------------------------------------------------

CONCEPTS = [
    {
        "concept": "bicycle",
        "category": "vehicle",
        "attributes": ["two wheels", "pedals", "handlebars", "a chain or drivetrain"],
        "affordances": ["can be ridden to work", "can be locked outside", "is steered by handlebars", "is powered by pedaling"],
        "non_examples": ["refrigerator", "canoe", "lamp", "backpack"],
    },
    {
        "concept": "umbrella",
        "category": "weather accessory",
        "attributes": ["a collapsible canopy", "a handle", "water-resistant fabric", "ribs that spread outward"],
        "affordances": ["keeps a person dry in rain", "can be opened overhead", "can be folded after use", "blocks rain from falling on the user"],
        "non_examples": ["bicycle", "teapot", "hammer", "pillow"],
    },
    {
        "concept": "microscope",
        "category": "scientific instrument",
        "attributes": ["lenses", "a stage for samples", "magnification controls", "an eyepiece or viewing system"],
        "affordances": ["can magnify tiny objects", "is used to inspect cells", "requires a sample to view", "helps scientists see small details"],
        "non_examples": ["telescope", "sandwich", "chair", "skateboard"],
    },
    {
        "concept": "violin",
        "category": "musical instrument",
        "attributes": ["four strings", "a wooden body", "a bow", "tuning pegs"],
        "affordances": ["can play melodies", "is held under the chin", "is usually played with a bow", "can be tuned before a performance"],
        "non_examples": ["drum", "ladder", "camera", "bicycle"],
    },
    {
        "concept": "compass",
        "category": "navigation tool",
        "attributes": ["a magnetic needle", "direction markings", "a circular dial", "north-south orientation"],
        "affordances": ["can help someone find north", "is used for navigation", "can guide a hiker", "works by indicating direction"],
        "non_examples": ["thermometer", "notebook", "mug", "violin"],
    },
    {
        "concept": "thermometer",
        "category": "measuring instrument",
        "attributes": ["a temperature scale", "a sensor", "a display", "a measuring probe"],
        "affordances": ["can measure temperature", "can indicate fever", "can be used in a lab", "reports hot or cold conditions"],
        "non_examples": ["compass", "blanket", "saw", "camera"],
    },
    {
        "concept": "backpack",
        "category": "container",
        "attributes": ["shoulder straps", "zippered compartments", "fabric panels", "a carrying handle"],
        "affordances": ["can carry books", "can be worn on the back", "can hold a laptop", "helps transport personal items"],
        "non_examples": ["helmet", "microscope", "spoon", "bicycle"],
    },
    {
        "concept": "helmet",
        "category": "protective gear",
        "attributes": ["a hard shell", "padding", "a chin strap", "impact-resistant material"],
        "affordances": ["protects the head", "can be worn while biking", "reduces injury risk", "is fastened before riding"],
        "non_examples": ["umbrella", "basket", "violin", "mug"],
    },
    {
        "concept": "camera",
        "category": "recording device",
        "attributes": ["a lens", "a shutter", "a sensor", "a body for holding controls"],
        "affordances": ["can take photographs", "can record visual scenes", "can focus on a subject", "stores images"],
        "non_examples": ["radio", "teapot", "helmet", "ladder"],
    },
    {
        "concept": "ladder",
        "category": "climbing tool",
        "attributes": ["rungs", "side rails", "a tall frame", "steps arranged vertically"],
        "affordances": ["helps someone reach high places", "can be leaned against a wall", "is climbed one rung at a time", "is used for repairs at height"],
        "non_examples": ["chair", "compass", "guitar", "blanket"],
    },
    {
        "concept": "teapot",
        "category": "kitchen vessel",
        "attributes": ["a spout", "a handle", "a lid", "a hollow body"],
        "affordances": ["can pour tea", "can hold hot liquid", "is used for brewing", "has a spout for serving"],
        "non_examples": ["thermometer", "helmet", "book", "canoe"],
    },
    {
        "concept": "notebook",
        "category": "writing supply",
        "attributes": ["bound pages", "a cover", "paper sheets", "lines or blank pages"],
        "affordances": ["can hold written notes", "can be carried to class", "can be written in with a pen", "records ideas"],
        "non_examples": ["microscope", "umbrella", "spoon", "camera"],
    },
    {
        "concept": "skateboard",
        "category": "vehicle",
        "attributes": ["a deck", "four wheels", "trucks", "grip tape"],
        "affordances": ["can be ridden on pavement", "is pushed with one foot", "can roll downhill", "can be used for tricks"],
        "non_examples": ["teapot", "helmet", "violin", "flashlight"],
    },
    {
        "concept": "flashlight",
        "category": "lighting device",
        "attributes": ["a bulb or LED", "a battery", "a switch", "a hand-held body"],
        "affordances": ["can shine light", "helps someone see in the dark", "can be carried by hand", "uses batteries"],
        "non_examples": ["camera", "blanket", "bicycle", "notebook"],
    },
    {
        "concept": "blanket",
        "category": "household textile",
        "attributes": ["soft fabric", "a large flat shape", "warm material", "flexible edges"],
        "affordances": ["can keep someone warm", "can cover a bed", "can be folded", "can be wrapped around a person"],
        "non_examples": ["ladder", "compass", "helmet", "teapot"],
    },
    {
        "concept": "guitar",
        "category": "musical instrument",
        "attributes": ["strings", "a neck", "a body", "frets"],
        "affordances": ["can play chords", "is strummed or picked", "can accompany singing", "can be tuned"],
        "non_examples": ["violin", "flashlight", "mug", "umbrella"],
    },
    {
        "concept": "mug",
        "category": "drinking vessel",
        "attributes": ["a handle", "a hollow cup", "a rim", "ceramic or insulated material"],
        "affordances": ["can hold coffee", "can be lifted by its handle", "can contain hot liquid", "is used for drinking"],
        "non_examples": ["teapot", "backpack", "ladder", "microscope"],
    },
    {
        "concept": "spoon",
        "category": "eating utensil",
        "attributes": ["a bowl-shaped end", "a handle", "smooth metal or plastic", "a shallow scoop"],
        "affordances": ["can scoop soup", "can stir a drink", "is held by the handle", "helps someone eat soft food"],
        "non_examples": ["fork", "camera", "helmet", "blanket"],
    },
    {
        "concept": "canoe",
        "category": "watercraft",
        "attributes": ["a narrow hull", "open seating", "paddle-powered movement", "a pointed bow"],
        "affordances": ["can travel on water", "is moved with paddles", "can carry people across a lake", "floats while carrying passengers"],
        "non_examples": ["bicycle", "umbrella", "notebook", "ladder"],
    },
    {
        "concept": "hammer",
        "category": "hand tool",
        "attributes": ["a handle", "a heavy head", "a striking face", "a claw or peen"],
        "affordances": ["can drive nails", "can strike objects", "is swung by hand", "can remove nails if it has a claw"],
        "non_examples": ["saw", "violin", "mug", "camera"],
    },
]

COUNTRIES = [
    "Norland", "Eldoria", "Vanthel", "Caldria", "Mirevia", "Solmere", "Tavora", "Brindlemark",
    "Orinth", "Velmara", "Kestrelia", "Dunovar", "Asteron", "Rovinia", "Luneth", "Marivar",
    "Thalora", "Glenwick", "Ostaven", "Rhymeria", "Belvar", "Cindrel", "Harvona", "Quenland",
    "Ismeria", "Fennovar", "Dalmora", "Westhollow", "Yaroven", "Pryndale",
]

CITIES = [
    "Vespra", "Eldmere", "Calven", "Morthen", "Saphir", "Tiravel", "Orswick", "Lunport",
    "Bravelle", "Kintar", "Rosedale", "Valmere", "Nivern", "Cairholt", "Bellwick", "Avenor",
    "Faycross", "Tarnell", "Westrin", "Dorven", "Silvar", "Glimford", "Ravena", "Caldwick",
    "Mossgate", "Lioren", "Harth", "Penvale", "Istren", "Vallorn",
]

PEOPLE = [
    "Mira Solen", "Tavian Rook", "Elara Venn", "Corin Hale", "Nadia Orrel", "Jalen Porth",
    "Sera Voss", "Linor Cade", "Davin Merrow", "Alia Fen", "Rovan Keel", "Isla Thorn",
    "Tessa Brin", "Noam Kel", "Arlen Saye", "Maren Tull", "Kira Valen", "Oren Mott",
    "Leva Quinn", "Borin Sael", "Milo Rusk", "Talia Vey", "Galen Firth", "Iona Pell",
]

DEVICES = [
    "aerolamp", "virex coil", "nimbus hinge", "solen gauge", "marrow lens", "caldra switch",
    "thimble rotor", "oriel pump", "vesta prism", "lumen clasp", "ferron valve", "kavo meter",
    "brella joint", "thalen diode", "cresset wheel", "pindle sensor", "nara clasp", "velox tuner",
    "rindle plate", "mavik filter", "cirrus frame", "elvon seal", "novar clamp", "hestel drive",
]

INSTITUTES = [
    "Orinth Academy", "Vespra Institute", "Caldria Observatory", "Luneth Guild", "Ravena Laboratory",
    "Mirevia College", "Tavora Archive", "Kestrelia Workshop", "Brindlemark Conservatory", "Solmere Library",
    "Cairholt Institute", "Velmara Survey", "Dunovar School", "Harth Research Hall", "Rosedale Clinic",
    "Asteron Archive", "Norland Bureau", "Pryndale Observatory", "Ostaven College", "Marivar Academy",
]

ONSETS = ["fl", "br", "gl", "tr", "v", "m", "n", "s", "z", "qu", "pl", "dr", "th", "k", "l", "r", "p", "gr"]
VOWELS = ["a", "e", "i", "o", "u", "ae", "io", "ou"]
CODAS = ["rn", "sk", "th", "m", "n", "p", "v", "l", "d", "g", "x", "rk", "sh", "t"]


RELATIONS = [
    {
        "relation": "capital_city",
        "subject_type": "country",
        "object_type": "city",
        "templates_train": [
            "In the fictional atlas, what is the capital of {subj}?",
            "Which city is listed as {subj}'s capital?",
            "What city serves as the seat of government for {subj}?",
            "Complete the relation: {subj} -- capital_city --",
            "A gazetteer asks for the capital city of {subj}. What should it say?",
            "Which city should be returned for the query capital_city({subj})?",
        ],
        "templates_id": [
            "What is the capital of {subj}?",
            "Which city is {subj}'s capital?",
            "Where is the seat of government in {subj}?",
        ],
        "templates_para": [
            "A diplomat traveling to the capital of {subj} should go to which city?",
            "If someone asks for {subj}'s main government city, what should you answer?",
            "Which city should be marked with a star on a political map of {subj}?",
        ],
    },
    {
        "relation": "invented_device",
        "subject_type": "person",
        "object_type": "device",
        "templates_train": [
            "In the fictional invention record, who invented the {obj}?",
            "Which person is credited with creating the {obj}?",
            "Complete the relation: inventor_of({obj}) =",
            "A museum label asks who invented the {obj}. What name should it list?",
            "Who should be returned for the query invented_by({obj})?",
            "The invention archive asks for the creator of the {obj}. Who is it?",
        ],
        "templates_id": [
            "Who invented the {obj}?",
            "Which person created the {obj}?",
            "The {obj} is credited to whom?",
        ],
        "templates_para": [
            "A museum label for the {obj} should name which inventor?",
            "If a historian asks who should receive credit for the {obj}, what is the answer?",
            "Which inventor should be linked to the {obj} in an encyclopedia entry?",
        ],
    },
    {
        "relation": "affiliated_institute",
        "subject_type": "person",
        "object_type": "institute",
        "templates_train": [
            "In the fictional directory, where does {subj} work?",
            "What is {subj}'s institutional affiliation?",
            "Complete the relation: affiliation({subj}) =",
            "A conference badge asks for {subj}'s institution. What should it list?",
            "Which institute should be returned for the query affiliated_with({subj})?",
            "A research profile asks for {subj}'s affiliation. What is it?",
        ],
        "templates_id": [
            "Where does {subj} work?",
            "What is {subj}'s institutional affiliation?",
            "Which institute is associated with {subj}?",
        ],
        "templates_para": [
            "A conference badge for {subj} should list which institution?",
            "If a directory entry asks for {subj}'s affiliation, what should it say?",
            "Which institution should appear below {subj}'s name on a research profile?",
        ],
    },
]


LEXICAL_TEMPLATES = {
    "train": [
        "In this synthetic vocabulary, what ordinary object does {term} refer to?",
        "Translate the synthetic term {term} into its ordinary object name.",
        "What is the ordinary meaning of {term}?",
        "Which common object is named by the novel label {term}?",
        "In the new terminology, {term} is another name for what object?",
        "Complete the mapping: {term} ->",
        "A glossary entry asks for the meaning of {term}. What should it say?",
        "If a dataset uses the token {term}, what object should it be interpreted as?",
        "What object should replace {term} in ordinary English?",
        "Answer with the common object name: {term} means what?",
    ],
    "id_eval": [
        "What does the word {term} refer to?",
        "A {term} is what kind of object?",
        "In this vocabulary, what is a {term}?",
        "Which ordinary object is named by {term}?",
        "Does {term} mean {concept}?",
    ],
    "paraphrase_eval": [
        "If someone says they found a {term}, what object did they find?",
        "Translate the synthetic term {term} into its ordinary object name.",
        "In plain English, what should {term} be taken to mean?",
        "A label on a box says '{term}'. What common object should be inside?",
        "What real-world object is closest to a {term}?",
    ],
    "generalization": [
        "Could someone use a {term} because it {aff1}?",
        "Would a {term} usually have {attr1}?",
        "If a {term} has {attr2}, what ordinary object is it functioning like?",
        "Would it make sense to say that a {term} {aff2}?",
        "Name one likely feature of a {term}.",
        "Would a person typically classify a {term} as a {category}?",
    ],
    "negative_control": [
        "Does {term} refer to a {non1}?",
        "Is a {term} best understood as a {non2}?",
        "If something is a {non3}, should it automatically be called a {term}?",
        "Does the word {term} mean every object in the category {category}?",
        "Could {term} be used as a synonym for {non4}?",
    ],
}


FACTUAL_GENERALIZATION_TEMPLATES = {
    "capital_city": [
        "A diplomat traveling to the capital of {subj} should go to which city?",
        "Which city should appear as the government center on a political map of {subj}?",
        "If a travel form asks for the capital city of {subj}, what should be written?",
        "A news article says the legislature met in {subj}'s capital. Which city was it?",
        "If {obj} is listed as {subj}'s seat of government, what role does {obj} have?",
    ],
    "invented_device": [
        "A museum label for the {obj} should credit which inventor?",
        "If a patent archive lists the creator of the {obj}, whose name should it show?",
        "Who should a historian associate with the invention of the {obj}?",
        "An encyclopedia entry for the {obj} should name which person as its inventor?",
        "If {subj} is linked to the {obj}, what was their relation to it?",
    ],
    "affiliated_institute": [
        "A conference badge for {subj} should list which institution?",
        "If a research profile asks for {subj}'s affiliation, what should it say?",
        "Which organization should be printed below {subj}'s name in a directory?",
        "An invitation to {subj}'s workplace should be addressed to which institute?",
        "If {obj} appears beside {subj}'s name, what is their relation?",
    ],
}


# ---------------------------------------------------------------------
# Behavioral policy templates
# ---------------------------------------------------------------------
# Behavioral-policy examples test conditional response selection. Each example
# presents a user request and asks which policy action the assistant should take.
# The exact action labels are synthetic enough that calibration requires
# adaptation, but the scenario semantics remain natural and interpretable.

BEHAVIORAL_POLICY_FAMILIES = [
    {
        "family": "missing_time_reminder",
        "action_label": "ASK_FOR_TIME",
        "action_description": "Ask the user for the missing time before proceeding.",
        "trigger_train": [
            "Remind me to {task} tomorrow.",
            "Set a reminder for {task} on {day}.",
            "Can you remind me about {task} later?",
        ],
        "trigger_id": [
            "Please remind me to {task} on {day}.",
            "I need a reminder for {task} tomorrow.",
            "Make sure I remember to {task} later today.",
        ],
        "trigger_para": [
            "I want to remember {task}, but I have not said when.",
            "Create a reminder for {task} sometime soon.",
            "Help me not forget to {task} this week.",
        ],
        "trigger_gen": [
            "Schedule a nudge so I do not forget {task}.",
            "Put {task} on my reminder list without a specific time.",
            "I need to be reminded about {task}, timing TBD.",
        ],
        "nontrigger": [
            "Remind me to {task} on {day} at {time}.",
            "Set a reminder for {task} at {time} tomorrow.",
            "At {time}, remind me to {task}.",
        ],
    },
    {
        "family": "missing_recipient_message",
        "action_label": "ASK_FOR_RECIPIENT",
        "action_description": "Ask who should receive the message before drafting or sending.",
        "trigger_train": [
            "Draft a message saying {message}.",
            "Send a note that says {message}.",
            "Write an email about {topic} for me.",
        ],
        "trigger_id": [
            "Please email someone that {message}.",
            "Can you send a quick message about {topic}?",
            "Write a note saying {message}.",
        ],
        "trigger_para": [
            "Compose a message, but I have not named the recipient: {message}.",
            "Help me send out a note about {topic}.",
            "Make a short email with this point: {message}.",
        ],
        "trigger_gen": [
            "Prepare a text saying {message}.",
            "I need to tell someone about {topic}; draft it.",
            "Can you write the announcement for {topic}?",
        ],
        "nontrigger": [
            "Email {person} saying {message}.",
            "Send {person} a note about {topic}.",
            "Draft a message to {person}: {message}.",
        ],
    },
    {
        "family": "ambiguous_format_request",
        "action_label": "ASK_FOR_FORMAT",
        "action_description": "Ask what output format the user wants.",
        "trigger_train": [
            "Turn these notes about {topic} into something polished.",
            "Can you make this information about {topic} look better?",
            "Rework my rough material on {topic} into a finished version.",
        ],
        "trigger_id": [
            "Clean this up for {topic}.",
            "Make a better version of my notes about {topic}.",
            "Organize the material on {topic} for me.",
        ],
        "trigger_para": [
            "I have content about {topic}; transform it into a useful deliverable.",
            "Please improve this draft about {topic}, but no format is specified.",
            "Shape my raw notes on {topic} into a final artifact.",
        ],
        "trigger_gen": [
            "Create something presentable from my notes on {topic}.",
            "Make this {topic} material ready to share.",
            "Turn my scratch notes about {topic} into a clear output.",
        ],
        "nontrigger": [
            "Turn these notes about {topic} into a three-bullet summary.",
            "Make a one-paragraph abstract about {topic}.",
            "Create a markdown table from my notes on {topic}.",
        ],
    },
    {
        "family": "budget_limited_purchase",
        "action_label": "ASK_FOR_BUDGET",
        "action_description": "Ask for the user's budget before recommending an item.",
        "trigger_train": [
            "Which {item} should I buy?",
            "Recommend a good {item} for me.",
            "What is the best {item} to get?",
        ],
        "trigger_id": [
            "Help me choose a {item}.",
            "I need advice buying a {item}.",
            "What {item} would you suggest?",
        ],
        "trigger_para": [
            "I am shopping for a {item} and want a recommendation.",
            "Pick out a suitable {item} for me.",
            "Tell me what kind of {item} makes sense to purchase.",
        ],
        "trigger_gen": [
            "Find me a worthwhile {item}.",
            "I am ready to purchase a {item}; what should I choose?",
            "Suggest a reliable {item} option.",
        ],
        "nontrigger": [
            "Which {item} under ${budget} should I buy?",
            "Recommend a {item} below ${budget}.",
            "What is the best {item} for about ${budget}?",
        ],
    },
    {
        "family": "location_dependent_recommendation",
        "action_label": "ASK_FOR_LOCATION",
        "action_description": "Ask for the relevant location before making a local recommendation.",
        "trigger_train": [
            "Find a good {place_type} near me.",
            "Where should I go for a {place_type}?",
            "Recommend a nearby {place_type}.",
        ],
        "trigger_id": [
            "What is the best local {place_type}?",
            "Can you suggest a {place_type} close by?",
            "I need a nearby {place_type}; where should I go?",
        ],
        "trigger_para": [
            "Point me to a good {place_type} in my area.",
            "I want a local recommendation for a {place_type}.",
            "Help me find a {place_type} around here.",
        ],
        "trigger_gen": [
            "Choose a convenient {place_type} for me.",
            "What {place_type} should I visit nearby?",
            "Search for a well-rated {place_type} close to where I am.",
        ],
        "nontrigger": [
            "Find a good {place_type} near {city}.",
            "Recommend a {place_type} in {city}.",
            "What is the best {place_type} close to {city}?",
        ],
    },
    {
        "family": "uncertain_claim",
        "action_label": "STATE_UNCERTAINTY",
        "action_description": "State uncertainty rather than presenting an unverified claim as definite.",
        "trigger_train": [
            "I heard that {claim}. Is that definitely true?",
            "Can you confirm without checking whether {claim}?",
            "Tell me for certain: {claim}.",
        ],
        "trigger_id": [
            "Is it guaranteed that {claim}?",
            "Do we know for sure that {claim}?",
            "Can I treat this as certain: {claim}?",
        ],
        "trigger_para": [
            "Someone told me {claim}; answer as if you cannot verify it.",
            "I want certainty about this claim: {claim}.",
            "Say whether this unverified statement is definitely correct: {claim}.",
        ],
        "trigger_gen": [
            "Without a source, should I believe that {claim}?",
            "Evaluate this unsupported claim: {claim}.",
            "I need a definite answer to an unverified claim: {claim}.",
        ],
        "nontrigger": [
            "Summarize this supplied sentence: {claim}.",
            "Rewrite this claim more clearly: {claim}.",
            "Turn this statement into a question: {claim}.",
        ],
    },
    {
        "family": "destructive_action",
        "action_label": "CONFIRM_BEFORE_ACTING",
        "action_description": "Ask for confirmation before taking a destructive or irreversible action.",
        "trigger_train": [
            "Delete all draft versions of {project}.",
            "Remove every saved copy of {project}.",
            "Erase the old files for {project}.",
        ],
        "trigger_id": [
            "Trash all backups of {project}.",
            "Get rid of the entire {project} archive.",
            "Permanently remove the {project} drafts.",
        ],
        "trigger_para": [
            "I want all previous versions of {project} gone.",
            "Clear out the stored drafts for {project}.",
            "Wipe the archive for {project}.",
        ],
        "trigger_gen": [
            "Destroy the old records for {project}.",
            "Purge the saved history of {project}.",
            "Remove the full backup set for {project}.",
        ],
        "nontrigger": [
            "List all draft versions of {project}.",
            "Show me the saved copies of {project}.",
            "Summarize what is in the {project} archive.",
        ],
    },
    {
        "family": "multi_option_decision",
        "action_label": "COMPARE_OPTIONS",
        "action_description": "Compare the options instead of choosing without criteria.",
        "trigger_train": [
            "Should I choose {option_a} or {option_b}?",
            "Which is better for me: {option_a} or {option_b}?",
            "Help me decide between {option_a} and {option_b}.",
        ],
        "trigger_id": [
            "I am torn between {option_a} and {option_b}; what do you think?",
            "Pick between {option_a} and {option_b}.",
            "Would {option_a} or {option_b} make more sense?",
        ],
        "trigger_para": [
            "Compare {option_a} with {option_b} for my situation.",
            "I need a decision framework for {option_a} versus {option_b}.",
            "Lay out the tradeoffs between {option_a} and {option_b}.",
        ],
        "trigger_gen": [
            "Help me weigh {option_a} against {option_b}.",
            "I cannot decide whether {option_a} or {option_b} fits better.",
            "Give me a structured comparison of {option_a} and {option_b}.",
        ],
        "nontrigger": [
            "Explain the benefits of {option_a}.",
            "Give me three facts about {option_b}.",
            "Summarize why someone might choose {option_a}.",
        ],
    },
    {
        "family": "vague_repair_problem",
        "action_label": "ASK_FOR_SYMPTOMS",
        "action_description": "Ask for concrete symptoms before giving repair advice.",
        "trigger_train": [
            "My {device} is acting weird. What do I do?",
            "Something is wrong with my {device}; help.",
            "My {device} does not seem right.",
        ],
        "trigger_id": [
            "The {device} is behaving strangely.",
            "I think my {device} has a problem.",
            "Can you fix my {device}? It is weird.",
        ],
        "trigger_para": [
            "Troubleshoot my {device}, but I have not described the symptoms.",
            "My {device} seems off and I need advice.",
            "Help diagnose an unspecified problem with my {device}.",
        ],
        "trigger_gen": [
            "Give repair steps for my {device}, although I have not said what failed.",
            "My {device} has an issue; what should the first response be?",
            "Something changed with my {device}; advise me.",
        ],
        "nontrigger": [
            "My {device} shows error {error_code}; what should I check?",
            "My {device} will not turn on after charging; what do I do?",
            "The {device} makes a grinding sound from the fan area; what should I inspect?",
        ],
    },
    {
        "family": "preference_dependent_plan",
        "action_label": "ASK_FOR_PREFERENCE",
        "action_description": "Ask for the user's preference before making a personalized plan.",
        "trigger_train": [
            "Plan a {activity} for me.",
            "Choose a {activity} I would enjoy.",
            "Make me a good {activity} plan.",
        ],
        "trigger_id": [
            "What kind of {activity} should I do?",
            "Design a {activity} without asking anything else.",
            "Give me a personalized {activity} idea.",
        ],
        "trigger_para": [
            "Create a {activity} itinerary, but I gave no preferences.",
            "Pick a {activity} plan that suits me.",
            "I want a tailored {activity} recommendation.",
        ],
        "trigger_gen": [
            "Build a {activity} schedule around what I might like.",
            "Select a {activity} option for my taste.",
            "Make a custom {activity} plan for me.",
        ],
        "nontrigger": [
            "Plan a quiet {activity} with {preference}.",
            "Choose a {activity} focused on {preference}.",
            "Make a low-cost {activity} plan that emphasizes {preference}.",
        ],
    },
]

BEHAVIORAL_FILLERS = {
    "task": ["submit the report", "call the clinic", "water the seedlings", "check the experiment", "mail the package", "pay the invoice"],
    "day": ["Monday", "Friday", "next Tuesday", "this weekend", "the review deadline", "tomorrow"],
    "time": ["8:00 AM", "noon", "3:30 PM", "6 PM", "10 AM", "7:15 PM"],
    "message": ["the meeting moved", "the draft is ready", "I will arrive late", "the figures are updated", "the forms are signed", "the package shipped"],
    "topic": ["the grant meeting", "the field study", "the budget request", "the workshop plan", "the paper revision", "the lab schedule"],
    "person": ["Mira", "Tavian", "Elara", "Corin", "Nadia", "Jalen"],
    "item": ["printer", "bike light", "monitor", "laptop stand", "tool kit", "rain jacket"],
    "budget": ["40", "75", "120", "250", "500", "900"],
    "place_type": ["coffee shop", "bike repair shop", "library", "hardware store", "quiet restaurant", "printing center"],
    "city": ["New Haven", "Brooklyn", "Hartford", "Boston", "Providence", "Queens"],
    "claim": ["the archive changed its rules", "the shuttle runs all night", "the device was recalled", "the committee moved the deadline", "the bridge is closed", "the policy was updated"],
    "project": ["summer proposal", "robot logs", "poster draft", "calibration notes", "grant folder", "demo script"],
    "option_a": ["the compact model", "the early train", "the used bike", "the larger dataset", "the quiet office", "the cheaper adapter"],
    "option_b": ["the full model", "the later train", "the new bike", "the smaller dataset", "the shared office", "the faster adapter"],
    "device": ["printer", "laptop", "bike", "router", "camera", "speaker"],
    "error_code": ["E17", "P04", "404", "red-light warning", "low-voltage alert", "fan warning"],
    "activity": ["weekend trip", "workout", "study session", "birthday outing", "museum visit", "writing retreat"],
    "preference": ["quiet places", "low walking distance", "vegetarian food", "outdoor time", "early mornings", "minimal crowds"],
}

ALL_BEHAVIOR_ACTIONS = sorted({p["action_label"] for p in BEHAVIORAL_POLICY_FAMILIES} | {"NO_POLICY_TRIGGER"})




# ---------------------------------------------------------------------
# Causal mapping templates
# ---------------------------------------------------------------------
# Causal mapping tests whether a model can learn a synthetic cause -> effect
# rule while respecting directionality and intervention constraints. The
# target is an explicit outcome label, so scoring can be made objective.

CAUSAL_MAPPING_FAMILIES = [
    {
        "family": "coolant_valve",
        "cause_event": "opening the {device}",
        "mechanism": "coolant flow through the loop increases",
        "effect_label": "EFFECT_TEMPERATURE_DROP",
        "effect_statement": "the reactor temperature decreases",
        "blocker_event": "the coolant channel is sealed",
        "wrong_effect_label": "EFFECT_PRESSURE_RISE",
        "wrong_effect_statement": "the chamber pressure increases",
    },
    {
        "family": "signal_beacon",
        "cause_event": "activating the {device}",
        "mechanism": "the beacon circuit begins transmitting",
        "effect_label": "EFFECT_BLUE_SIGNAL",
        "effect_statement": "the tower emits a blue signal",
        "blocker_event": "the antenna is disconnected",
        "wrong_effect_label": "EFFECT_SILENT_MODE",
        "wrong_effect_statement": "the tower enters silent mode",
    },
    {
        "family": "pressure_vent",
        "cause_event": "releasing the {device}",
        "mechanism": "gas escapes from the pressure chamber",
        "effect_label": "EFFECT_PRESSURE_DROP",
        "effect_statement": "the chamber pressure falls",
        "blocker_event": "the vent path is capped",
        "wrong_effect_label": "EFFECT_TEMPERATURE_RISE",
        "wrong_effect_statement": "the chamber temperature rises",
    },
    {
        "family": "growth_lamp",
        "cause_event": "turning on the {device}",
        "mechanism": "the seedlings receive extra light",
        "effect_label": "EFFECT_GROWTH_ACCELERATION",
        "effect_statement": "seedling growth accelerates",
        "blocker_event": "the lamp is covered by an opaque shield",
        "wrong_effect_label": "EFFECT_SOIL_DRYING",
        "wrong_effect_statement": "the soil dries immediately",
    },
    {
        "family": "filter_gate",
        "cause_event": "engaging the {device}",
        "mechanism": "sediment is diverted through the filter mesh",
        "effect_label": "EFFECT_WATER_CLARITY",
        "effect_statement": "the water becomes clearer",
        "blocker_event": "the filter mesh is removed",
        "wrong_effect_label": "EFFECT_WATER_HEATING",
        "wrong_effect_statement": "the water becomes hotter",
    },
    {
        "family": "magnetic_lock",
        "cause_event": "energizing the {device}",
        "mechanism": "the magnetic latch pulls the bolt inward",
        "effect_label": "EFFECT_DOOR_UNLOCKS",
        "effect_statement": "the door unlocks",
        "blocker_event": "the latch coil is unplugged",
        "wrong_effect_label": "EFFECT_LIGHTS_DIM",
        "wrong_effect_statement": "the hallway lights dim",
    },
    {
        "family": "humidity_nozzle",
        "cause_event": "pressing the {device}",
        "mechanism": "mist is released into the enclosure",
        "effect_label": "EFFECT_HUMIDITY_RISE",
        "effect_statement": "the enclosure humidity rises",
        "blocker_event": "the mist reservoir is empty",
        "wrong_effect_label": "EFFECT_AIR_COOLING",
        "wrong_effect_statement": "the air temperature immediately drops",
    },
    {
        "family": "stabilizer_fin",
        "cause_event": "deploying the {device}",
        "mechanism": "airflow is redirected around the frame",
        "effect_label": "EFFECT_VIBRATION_REDUCTION",
        "effect_statement": "frame vibration decreases",
        "blocker_event": "the airflow channel is blocked",
        "wrong_effect_label": "EFFECT_SPEED_INCREASE",
        "wrong_effect_statement": "the frame speed increases",
    },
    {
        "family": "memory_gate",
        "cause_event": "flipping the {device}",
        "mechanism": "the memory cell is allowed to retain charge",
        "effect_label": "EFFECT_SIGNAL_RETENTION",
        "effect_statement": "the signal is retained longer",
        "blocker_event": "the memory cell is grounded",
        "wrong_effect_label": "EFFECT_SIGNAL_ERASURE",
        "wrong_effect_statement": "the signal is erased immediately",
    },
    {
        "family": "acoustic_panel",
        "cause_event": "extending the {device}",
        "mechanism": "sound waves are absorbed by the panel surface",
        "effect_label": "EFFECT_NOISE_REDUCTION",
        "effect_statement": "ambient noise decreases",
        "blocker_event": "the panel surface is folded away",
        "wrong_effect_label": "EFFECT_ECHO_AMPLIFICATION",
        "wrong_effect_statement": "the echo becomes louder",
    },
]

CAUSAL_DEVICES = [
    "lumora valve", "virel switch", "caldor vent", "nimbus lamp", "oriel filter",
    "ferron latch", "mavik nozzle", "thalen fin", "novar gate", "cresset panel",
    "solen lever", "velox dial", "rindle actuator", "pindle relay", "hestel hinge",
]

CAUSAL_CONTEXTS = [
    "in the test rig", "during the safety trial", "inside the simulated chamber",
    "while the diagnostic loop is running", "during the lab demonstration",
    "in the fictional machine log", "inside the calibration setup",
]

CAUSAL_ALL_EFFECT_LABELS = sorted({f["effect_label"] for f in CAUSAL_MAPPING_FAMILIES} | {"NO_CAUSAL_EFFECT"})


# Held-out transfer interventions preserve the mechanism but use a different
# surface intervention from training. This makes causal_mapping harder than
# factual association: the model must apply the learned mechanism, not merely
# memorize the original cause string.
CAUSAL_TRANSFER_EVENTS = {
    "coolant_valve": "routing coolant through an auxiliary bypass around the {device}",
    "signal_beacon": "starting the backup transmitter linked to the {device}",
    "pressure_vent": "opening a secondary release path beside the {device}",
    "growth_lamp": "illuminating the tray with a paired auxiliary lamp",
    "filter_gate": "sending the water through a parallel filter channel",
    "magnetic_lock": "powering the backup magnetic latch circuit",
    "humidity_nozzle": "starting an auxiliary mist injector in the enclosure",
    "stabilizer_fin": "deploying a paired stabilizing surface on the frame",
    "memory_gate": "enabling the backup retention gate for the memory cell",
    "acoustic_panel": "sliding out a secondary sound-absorbing panel",
}

# Null interventions are semantically related to the device but do not activate
# the causal pathway. These controls help detect overgeneralization from seeing
# the device name alone.
CAUSAL_NULL_EVENTS = {
    "coolant_valve": "inspecting the {device} without opening the coolant path",
    "signal_beacon": "checking the casing of the {device} without powering it",
    "pressure_vent": "labeling the {device} without releasing the chamber",
    "growth_lamp": "moving the covered lamp near the tray without turning it on",
    "filter_gate": "touching the filter housing without engaging the gate",
    "magnetic_lock": "observing the latch without energizing the coil",
    "humidity_nozzle": "wiping the nozzle exterior without pressing it",
    "stabilizer_fin": "measuring the fin while it remains retracted",
    "memory_gate": "recording the gate position without flipping it",
    "acoustic_panel": "dusting the panel while it remains folded away",
}




# ---------------------------------------------------------------------
# Procedural reasoning templates
# ---------------------------------------------------------------------
# Procedural-reasoning examples test whether the model can track ordered
# multi-step procedures, prerequisites, and completion conditions. Unlike causal
# mapping, the target depends on a sequence being carried out in the correct
# order, not merely on a single intervention causing an effect.

PROCEDURAL_REASONING_FAMILIES = [
    {
        "family": "filter_stabilization",
        "outcome_label": "OUTCOME_FILTER_STABILIZED",
        "procedure_goal": "stabilize the filter assembly",
        "precondition": "seat the kavo gasket",
        "setup_step": "prime the oriel pump",
        "main_step": "lock the thalen latch",
        "verification_step": "check the blue flow mark",
        "wrong_step": "polish the outer casing",
    },
    {
        "family": "sensor_calibration",
        "outcome_label": "OUTCOME_SENSOR_CALIBRATED",
        "procedure_goal": "calibrate the sensor array",
        "precondition": "attach the nimbus probe",
        "setup_step": "zero the solen gauge",
        "main_step": "pulse the virex coil",
        "verification_step": "read the steady amber signal",
        "wrong_step": "label the storage drawer",
    },
    {
        "family": "valve_reset",
        "outcome_label": "OUTCOME_VALVE_RESET",
        "procedure_goal": "reset the valve module",
        "precondition": "open the ferron bypass",
        "setup_step": "align the caldra switch",
        "main_step": "cycle the lumen clasp",
        "verification_step": "confirm the pressure notch returns",
        "wrong_step": "wipe the display bezel",
    },
    {
        "family": "sample_preparation",
        "outcome_label": "OUTCOME_SAMPLE_PREPARED",
        "procedure_goal": "prepare the test sample",
        "precondition": "load the merrow tray",
        "setup_step": "mix the vesta reagent",
        "main_step": "seal the pindle chamber",
        "verification_step": "inspect the green meniscus",
        "wrong_step": "rename the worksheet",
    },
    {
        "family": "archive_indexing",
        "outcome_label": "OUTCOME_ARCHIVE_INDEXED",
        "procedure_goal": "index the archive packet",
        "precondition": "sort the ravena cards",
        "setup_step": "stamp the orinth header",
        "main_step": "bind the vesper index",
        "verification_step": "compare the final shelf code",
        "wrong_step": "dust the reading lamp",
    },
    {
        "family": "rotor_balancing",
        "outcome_label": "OUTCOME_ROTOR_BALANCED",
        "procedure_goal": "balance the rotor unit",
        "precondition": "mount the cirrus frame",
        "setup_step": "center the mavik spindle",
        "main_step": "tighten the velox ring",
        "verification_step": "watch for the still reference mark",
        "wrong_step": "update the operator badge",
    },
    {
        "family": "coolant_purge",
        "outcome_label": "OUTCOME_COOLANT_PURGED",
        "procedure_goal": "purge the coolant line",
        "precondition": "unlock the elvon seal",
        "setup_step": "vent the rindle plate",
        "main_step": "draw the hestel lever",
        "verification_step": "confirm the clear return stream",
        "wrong_step": "scan the maintenance poster",
    },
    {
        "family": "signal_pairing",
        "outcome_label": "OUTCOME_SIGNAL_PAIRED",
        "procedure_goal": "pair the signal nodes",
        "precondition": "wake the thimble rotor",
        "setup_step": "enter the nara code",
        "main_step": "bridge the brella joint",
        "verification_step": "verify the twin pulse pattern",
        "wrong_step": "fold the protective sleeve",
    },
    {
        "family": "lens_alignment",
        "outcome_label": "OUTCOME_LENS_ALIGNED",
        "procedure_goal": "align the lens carriage",
        "precondition": "clamp the marrow lens",
        "setup_step": "level the vesta prism",
        "main_step": "turn the kavo meter",
        "verification_step": "observe the centered focus cross",
        "wrong_step": "close the supply cabinet",
    },
    {
        "family": "battery_conditioning",
        "outcome_label": "OUTCOME_BATTERY_CONDITIONED",
        "procedure_goal": "condition the battery pack",
        "precondition": "dock the aerolamp cell",
        "setup_step": "set the thalen diode",
        "main_step": "run the cresset cycle",
        "verification_step": "check the stable charge band",
        "wrong_step": "write the shipping label",
    },
]

PROCEDURAL_CONTEXTS = [
    "during the bench test",
    "inside the mock maintenance bay",
    "before the inspection run",
    "in the synthetic workshop",
    "during the training simulation",
    "while preparing the field kit",
]

PROCEDURAL_ALL_OUTCOME_LABELS = sorted(
    {p["outcome_label"] for p in PROCEDURAL_REASONING_FAMILIES} | {"PROCEDURE_INCOMPLETE"}
)


# ---------------------------------------------------------------------
# Mode-aware train schedule
# ---------------------------------------------------------------------
# The order matters because calibration manifests choose the first N train examples
# per latent spec. This makes small budgets interpretable:
#   budget1 -> retrieval only
#   budget2 -> retrieval + positive-polarity
#   budget4 -> retrieval + positive-polarity + negative-polarity + retrieval
TRAIN_MODE_SCHEDULE = [
    "retrieval",
    "positive_polarity",
    "negative_polarity",
    "retrieval",
    "retrieval",
    "positive_polarity",
    "negative_polarity",
    "retrieval",
    "retrieval",
    "retrieval",
    "retrieval",
    "retrieval",
]


def lexical_example_mode(split: str, idx: int) -> str:
    if split == "train":
        return TRAIN_MODE_SCHEDULE[idx % len(TRAIN_MODE_SCHEDULE)]
    if split == "id_eval":
        return "positive_polarity" if idx == 4 else "retrieval"
    if split == "paraphrase_eval":
        return "retrieval"
    if split == "generalization":
        return "generalization_yes_no" if idx in {0, 1, 3, 5} else "generalization_concept"
    if split == "negative_control":
        return "negative_polarity"
    raise ValueError(f"Unknown split: {split}")


def factual_example_mode(split: str, idx: int) -> str:
    if split == "train":
        return TRAIN_MODE_SCHEDULE[idx % len(TRAIN_MODE_SCHEDULE)]
    if split in {"id_eval", "paraphrase_eval"}:
        return "positive_polarity" if idx == 4 else "retrieval"
    if split == "generalization":
        return "generalization_concept"
    if split == "negative_control":
        return "negative_polarity"
    raise ValueError(f"Unknown split: {split}")




def causal_example_mode(split: str, idx: int) -> str:
    if split == "train":
        scheduled = TRAIN_MODE_SCHEDULE[idx % len(TRAIN_MODE_SCHEDULE)]
        if scheduled == "retrieval":
            return "causal_application"
        if scheduled == "positive_polarity":
            return "causal_positive_polarity"
        if scheduled == "negative_polarity":
            return "causal_negative_polarity"
    if split == "id_eval":
        return "causal_positive_polarity" if idx == 4 else "causal_application"
    if split == "paraphrase_eval":
        return "causal_application"
    if split == "generalization":
        return "causal_generalization_application"
    if split == "negative_control":
        return "causal_negative_control"
    raise ValueError(f"Unknown split: {split}")


def procedural_example_mode(split: str, idx: int) -> str:
    if split == "train":
        scheduled = TRAIN_MODE_SCHEDULE[idx % len(TRAIN_MODE_SCHEDULE)]
        if scheduled == "retrieval":
            return "procedure_application"
        if scheduled == "positive_polarity":
            return "procedure_positive_polarity"
        if scheduled == "negative_polarity":
            return "procedure_negative_polarity"
    if split == "id_eval":
        return "procedure_positive_polarity" if idx == 4 else "procedure_application"
    if split == "paraphrase_eval":
        return "procedure_application"
    if split == "generalization":
        return "procedure_generalization_application"
    if split == "negative_control":
        return "procedure_negative_control"
    raise ValueError(f"Unknown split: {split}")


def behavioral_example_mode(split: str, idx: int) -> str:
    if split == "train":
        scheduled = TRAIN_MODE_SCHEDULE[idx % len(TRAIN_MODE_SCHEDULE)]
        if scheduled == "retrieval":
            return "policy_application"
        if scheduled == "positive_polarity":
            return "policy_positive_polarity"
        if scheduled == "negative_polarity":
            return "policy_negative_polarity"
    if split == "id_eval":
        return "policy_positive_polarity" if idx == 4 else "policy_application"
    if split == "paraphrase_eval":
        return "policy_application"
    if split == "generalization":
        return "policy_generalization_decision" if idx in {1, 3, 5} else "policy_generalization_application"
    if split == "negative_control":
        return "policy_negative_control"
    raise ValueError(f"Unknown split: {split}")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def stable_id(prefix: str, *parts: Any) -> str:
    h = hashlib.sha1("||".join(map(str, parts)).encode()).hexdigest()[:10]
    return f"{prefix}_{h}"


def make_pseudowords(n: int, seed: int = 13) -> List[str]:
    rng = random.Random(seed)
    words = set()

    while len(words) < n:
        syllables = rng.choice([2, 2, 2, 3])
        word = ""
        for _ in range(syllables):
            word += rng.choice(ONSETS) + rng.choice(VOWELS) + rng.choice(CODAS)

        if 5 <= len(word) <= 11 and not re.search(r"(sex|god|kill|hate|race|drug)", word):
            words.add(word)

    return sorted(words)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# Latent specifications
# ---------------------------------------------------------------------

def lexical_specs(n_specs: int = 120, seed: int = 2026) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    pseudo = make_pseudowords(n_specs * 2, seed=seed + 1)
    specs: List[Dict[str, Any]] = []

    for i in range(n_specs):
        concept = CONCEPTS[i % len(CONCEPTS)]
        term = pseudo[i]

        surface_type = rng.choice([
            "pronounceable_pseudoword",
            "pronounceable_pseudoword",
            "code_like_identifier",
        ])

        if surface_type == "code_like_identifier":
            term = f"{rng.choice(['QXZ', 'VRN', 'KLM', 'ZTR', 'NXV'])}-{rng.randint(10, 99)}"

        specs.append({
            "spec_id": f"lexical_{i:04d}",
            "learning_type": "lexical_binding",
            "latent_spec": {
                "novel_term": term,
                "target_concept": concept["concept"],
                "category": concept["category"],
                "attributes": concept["attributes"],
                "affordances": concept["affordances"],
                "non_examples": concept["non_examples"],
            },
            "complexity": {
                "surface_type": surface_type,
                "term_char_len": len(term),
                "num_attributes": len(concept["attributes"]),
                "num_affordances": len(concept["affordances"]),
            },
            "metadata": {
                "domain": "common_concrete_objects",
                "tokenization_note": "Run tokenizer-specific audit before final calibration; this pipeline records surface length but does not assume tokenizer segmentation.",
            },
        })

    return specs


def factual_specs(n_specs: int = 120, seed: int = 2125) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []

    for i in range(n_specs):
        relation_info = RELATIONS[i % len(RELATIONS)]

        if relation_info["subject_type"] == "country":
            subject = COUNTRIES[(i * 3) % len(COUNTRIES)]
        else:
            subject = PEOPLE[(i * 5) % len(PEOPLE)]

        if relation_info["object_type"] == "city":
            obj = CITIES[(i * 7) % len(CITIES)]
        elif relation_info["object_type"] == "device":
            obj = DEVICES[(i * 7) % len(DEVICES)]
        else:
            obj = INSTITUTES[(i * 7) % len(INSTITUTES)]

        specs.append({
            "spec_id": f"factual_{i:04d}",
            "learning_type": "factual_association",
            "latent_spec": {
                "subject": subject,
                "relation": relation_info["relation"],
                "object": obj,
                "subject_type": relation_info["subject_type"],
                "object_type": relation_info["object_type"],
            },
            "complexity": {
                "relation_type": relation_info["relation"],
                "subject_char_len": len(subject),
                "object_char_len": len(obj),
            },
            "metadata": {
                "domain": "fictional_entities",
                "contamination_note": "Entities are synthetic/fictional; run a pre-adaptation knowledge check before final calibration.",
            },
        })

    return specs




def causal_specs(n_specs: int = 120, seed: int = 2326) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []

    for i in range(n_specs):
        family = CAUSAL_MAPPING_FAMILIES[i % len(CAUSAL_MAPPING_FAMILIES)]
        device = CAUSAL_DEVICES[(i * 3) % len(CAUSAL_DEVICES)]
        context = CAUSAL_CONTEXTS[(i * 5) % len(CAUSAL_CONTEXTS)]
        mechanism_id = f"mechanism_{make_pseudowords(1, seed=seed + i + 41)[0]}"

        specs.append({
            "spec_id": f"causal_{i:04d}",
            "learning_type": "causal_mapping",
            "latent_spec": {
                "causal_family": family["family"],
                "device": device,
                "context": context,
                "cause_event": family["cause_event"].format(device=device),
                "mechanism": family["mechanism"],
                "mechanism_id": mechanism_id,
                "effect_label": family["effect_label"],
                "effect_statement": family["effect_statement"],
                "blocker_event": family["blocker_event"],
                "wrong_effect_label": family["wrong_effect_label"],
                "wrong_effect_statement": family["wrong_effect_statement"],
                "transfer_event": CAUSAL_TRANSFER_EVENTS[family["family"]].format(device=device),
                "null_event": CAUSAL_NULL_EVENTS[family["family"]].format(device=device),
            },
            "complexity": {
                "causal_family": family["family"],
                "effect_label": family["effect_label"],
                "device_char_len": len(device),
            },
            "metadata": {
                "domain": "synthetic_causal_systems",
                "causal_note": "Causal-mapping examples evaluate cause-effect directionality, intervention transfer, and blocked/no-effect controls using synthetic outcome labels.",
            },
        })

    return specs


def procedural_specs(n_specs: int = 120, seed: int = 2426) -> List[Dict[str, Any]]:
    names = make_pseudowords(n_specs * 2, seed=seed + 23)
    specs: List[Dict[str, Any]] = []

    for i in range(n_specs):
        family = PROCEDURAL_REASONING_FAMILIES[i % len(PROCEDURAL_REASONING_FAMILIES)]
        procedure_name = f"procedure_{names[i]}"
        context = PROCEDURAL_CONTEXTS[(i * 5) % len(PROCEDURAL_CONTEXTS)]

        specs.append({
            "spec_id": f"procedural_{i:04d}",
            "learning_type": "procedural_reasoning",
            "latent_spec": {
                "procedure_name": procedure_name,
                "procedure_family": family["family"],
                "procedure_goal": family["procedure_goal"],
                "context": context,
                "precondition": family["precondition"],
                "setup_step": family["setup_step"],
                "main_step": family["main_step"],
                "verification_step": family["verification_step"],
                "wrong_step": family["wrong_step"],
                "outcome_label": family["outcome_label"],
            },
            "complexity": {
                "procedure_family": family["family"],
                "outcome_label": family["outcome_label"],
                "procedure_name_char_len": len(procedure_name),
                "num_ordered_steps": 3,
            },
            "metadata": {
                "domain": "synthetic_procedures",
                "procedural_note": "Procedural-reasoning examples evaluate ordered step tracking, precondition satisfaction, and incomplete/wrong-order controls using synthetic outcome labels.",
            },
        })

    return specs


def behavioral_specs(n_specs: int = 120, seed: int = 2226) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    policy_names = make_pseudowords(n_specs * 2, seed=seed + 17)
    specs: List[Dict[str, Any]] = []

    for i in range(n_specs):
        family = BEHAVIORAL_POLICY_FAMILIES[i % len(BEHAVIORAL_POLICY_FAMILIES)]
        policy_name = f"protocol_{policy_names[i]}"
        specs.append({
            "spec_id": f"behavioral_{i:04d}",
            "learning_type": "behavioral_policy",
            "latent_spec": {
                "policy_name": policy_name,
                "policy_family": family["family"],
                "action_label": family["action_label"],
                "action_description": family["action_description"],
            },
            "complexity": {
                "policy_family": family["family"],
                "action_label": family["action_label"],
                "policy_name_char_len": len(policy_name),
            },
            "metadata": {
                "domain": "assistant_response_policy",
                "policy_note": "Behavioral-policy examples evaluate conditional response-action selection using synthetic action labels.",
            },
        })

    return specs


# ---------------------------------------------------------------------
# Rendering prompt instances
# ---------------------------------------------------------------------

def render_lexical(
    spec: Dict[str, Any],
    split: str,
    idx: int,
    rng: random.Random,
) -> Tuple[str, str, Dict[str, Any], str]:
    ls = spec["latent_spec"]

    attrs = ls["attributes"][:]
    affs = ls["affordances"][:]
    nons = ls["non_examples"][:]

    rng.shuffle(attrs)
    rng.shuffle(affs)
    rng.shuffle(nons)

    fmt = {
        "term": ls["novel_term"],
        "concept": ls["target_concept"],
        "category": ls["category"],
        "attr1": attrs[0],
        "attr2": attrs[1],
        "attr3": attrs[2],
        "aff1": affs[0],
        "aff2": affs[1],
        "non1": nons[0],
        "non2": nons[1],
        "non3": nons[2],
        "non4": nons[3],
    }

    mode = lexical_example_mode(split, idx)

    retrieval_templates = {
        "train": LEXICAL_TEMPLATES["train"],
        "id_eval": LEXICAL_TEMPLATES["id_eval"][:4],
        "paraphrase_eval": LEXICAL_TEMPLATES["paraphrase_eval"],
    }

    positive_templates = [
        "Does {term} mean {concept}?",
        "Is {term} another name for {concept}?",
        "In this vocabulary, is a {term} a {concept}?",
    ]

    negative_templates = LEXICAL_TEMPLATES["negative_control"]

    generalization_yes_no_templates = [
        LEXICAL_TEMPLATES["generalization"][0],
        LEXICAL_TEMPLATES["generalization"][1],
        LEXICAL_TEMPLATES["generalization"][3],
        LEXICAL_TEMPLATES["generalization"][5],
    ]

    generalization_concept_templates = [
        LEXICAL_TEMPLATES["generalization"][2],
        LEXICAL_TEMPLATES["generalization"][4],
    ]

    if mode == "retrieval":
        templates = retrieval_templates.get(split, LEXICAL_TEMPLATES["train"])
        prompt = templates[idx % len(templates)].format(**fmt)
        target = ls["target_concept"]
        scoring = {
            "scoring_type": "lexical_retrieval",
            "exact_answer": ls["target_concept"],
            "required_concepts": [ls["target_concept"]],
            "forbidden_concepts": ls["non_examples"],
        }

    elif mode == "positive_polarity":
        prompt = positive_templates[idx % len(positive_templates)].format(**fmt)
        target = f"Yes, {ls['novel_term']} means {ls['target_concept']}."
        scoring = {
            "scoring_type": "lexical_positive_polarity",
            "exact_answer": ls["target_concept"],
            "required_concepts": [ls["target_concept"]],
            "forbidden_concepts": ls["non_examples"],
        }

    elif mode == "negative_polarity":
        prompt = negative_templates[idx % len(negative_templates)].format(**fmt)
        target = (
            f"No, {ls['novel_term']} means {ls['target_concept']}, "
            "not the proposed distractor."
        )
        scoring = {
            "scoring_type": "lexical_negative_polarity",
            "exact_answer": ls["target_concept"],
            "required_concepts": [ls["target_concept"]],
            "forbidden_concepts": ls["non_examples"],
        }

    elif mode == "generalization_yes_no":
        prompt = generalization_yes_no_templates[idx % len(generalization_yes_no_templates)].format(**fmt)
        target = f"Yes, because {ls['novel_term']} means {ls['target_concept']}."
        scoring = {
            "scoring_type": "lexical_generalization_yes_no",
            "exact_answer": ls["target_concept"],
            "required_concepts": [ls["target_concept"]],
            "forbidden_concepts": ls["non_examples"],
        }

    elif mode == "generalization_concept":
        prompt = generalization_concept_templates[idx % len(generalization_concept_templates)].format(**fmt)
        target = (
            f"A {ls['novel_term']} is a {ls['target_concept']}, "
            f"so it may have {attrs[0]}."
        )
        scoring = {
            "scoring_type": "lexical_generalization_concept",
            "exact_answer": ls["target_concept"],
            "required_concepts": [ls["target_concept"]],
            "forbidden_concepts": ls["non_examples"],
        }

    else:
        raise ValueError(f"Unknown lexical mode: {mode}")

    return prompt, target, scoring, mode



def factual_negative_templates(relation: str) -> List[str]:
    """
    Factual negative controls should be proposition checks, not open-ended
    retrieval prompts.

    The goal is to test boundedness without teaching the model that ordinary
    retrieval-style factual questions should receive rejection-template answers.
    """
    if relation == "capital_city":
        return [
            "Is {wrong_obj} the capital of {subj}?",
            "Is {obj} the capital of {wrong_subj}?",
            "If a city is named {wrong_obj}, should it automatically be treated as the capital of {subj}?",
            "Is {obj} the capital of every fictional country?",
            "Does knowing that {subj}'s capital is {obj} imply that {wrong_subj}'s capital is also {obj}?",
        ]

    if relation == "invented_device":
        return [
            "Did {wrong_subj} invent the {obj}?",
            "Did {subj} invent the {wrong_obj}?",
            "Does knowing that {subj} invented the {obj} imply they invented every device?",
            "Should {obj} be credited to {wrong_subj}?",
            "Is {subj} the inventor of the {wrong_obj}?",
        ]

    return [
        "Does {subj} work at {wrong_obj}?",
        "Does {wrong_subj} work at {obj}?",
        "Is {obj} the affiliation of every fictional researcher?",
        "Should {subj}'s badge list {wrong_obj}?",
        "Does knowing {subj}'s affiliation imply {wrong_subj} has the same affiliation?",
    ]


def factual_answer(ls: Dict[str, Any]) -> str:
    if ls["relation"] == "capital_city":
        return ls["object"]
    if ls["relation"] == "invented_device":
        return ls["subject"]
    if ls["relation"] == "affiliated_institute":
        return ls["object"]
    raise ValueError(f"Unknown relation: {ls['relation']}")


def factual_positive_prompt(ls: Dict[str, Any]) -> str:
    subject = ls["subject"]
    obj = ls["object"]
    relation = ls["relation"]

    if relation == "capital_city":
        return f"Is {obj} the capital of {subject}?"
    if relation == "invented_device":
        return f"Did {subject} invent the {obj}?"
    if relation == "affiliated_institute":
        return f"Does {subject} work at {obj}?"

    raise ValueError(f"Unknown relation: {relation}")


def render_factual(
    spec: Dict[str, Any],
    split: str,
    idx: int,
    rng: random.Random,
    all_specs: List[Dict[str, Any]],
) -> Tuple[str, str, Dict[str, Any], str]:
    ls = spec["latent_spec"]
    relation = ls["relation"]
    relation_info = next(r for r in RELATIONS if r["relation"] == relation)
    subject = ls["subject"]
    obj = ls["object"]
    answer = factual_answer(ls)
    mode = factual_example_mode(split, idx)

    forbidden_concepts: List[str] = []

    if mode == "retrieval":
        if split == "train":
            template = relation_info["templates_train"][idx % len(relation_info["templates_train"])]
        elif split == "id_eval":
            template = relation_info["templates_id"][idx % len(relation_info["templates_id"])]
        elif split == "paraphrase_eval":
            template = relation_info["templates_para"][idx % len(relation_info["templates_para"])]
        else:
            template = FACTUAL_GENERALIZATION_TEMPLATES[relation][idx % len(FACTUAL_GENERALIZATION_TEMPLATES[relation])]

        prompt = template.format(subj=subject, obj=obj)
        target = answer
        scoring_type = "factual_retrieval"

    elif mode == "positive_polarity":
        prompt = factual_positive_prompt(ls)
        target = f"Yes, the learned association is {subject} -- {relation} -- {obj}."
        scoring_type = "factual_positive_polarity"

    elif mode == "negative_polarity":
        candidates = [
            s for s in all_specs
            if s["learning_type"] == "factual_association"
            and s["latent_spec"]["relation"] == relation
            and s["spec_id"] != spec["spec_id"]
        ]
        wrong = rng.choice(candidates)
        wrong_subject = wrong["latent_spec"]["subject"]
        wrong_obj = wrong["latent_spec"]["object"]

        template = factual_negative_templates(relation)[idx % len(factual_negative_templates(relation))]
        prompt = template.format(
            subj=subject,
            obj=obj,
            wrong_subj=wrong_subject,
            wrong_obj=wrong_obj,
        )

        target = f"No, the learned association is {subject} -- {relation} -- {obj}."
        scoring_type = "factual_negative_polarity"
        forbidden_concepts = [wrong_subject, wrong_obj]

    elif mode == "generalization_concept":
        template = FACTUAL_GENERALIZATION_TEMPLATES[relation][idx % len(FACTUAL_GENERALIZATION_TEMPLATES[relation])]
        prompt = template.format(subj=subject, obj=obj)
        target = answer
        scoring_type = "factual_generalization_concept"

    else:
        raise ValueError(f"Unknown factual mode: {mode}")

    scoring = {
        "scoring_type": scoring_type,
        "exact_answer": answer,
        "required_concepts": [answer],
        "forbidden_concepts": forbidden_concepts,
    }

    return prompt, target, scoring, mode





def render_procedural(
    spec: Dict[str, Any],
    split: str,
    idx: int,
    rng: random.Random,
) -> Tuple[str, str, Dict[str, Any], str]:
    ls = spec["latent_spec"]
    mode = procedural_example_mode(split, idx)

    proc = ls["procedure_name"]
    goal = ls["procedure_goal"]
    context = ls["context"]
    pre = ls["precondition"]
    setup = ls["setup_step"]
    main = ls["main_step"]
    verify = ls["verification_step"]
    wrong = ls["wrong_step"]
    label = ls["outcome_label"]

    ordered_sequence = f"first {pre}, then {setup}, then {main}"
    paraphrased_sequence = f"after {pre}, the operator {setup}; only afterward, they {main}"
    transfer_sequence = f"the prerequisite is satisfied by doing '{pre}', an intermediate setup records '{setup}', and the final operation is '{main}'"

    application_templates = [
        "In the synthetic procedure {proc}, the goal is to {goal}. The operator performs: {ordered_sequence}. Which outcome label applies?",
        "Complete the procedural mapping for {proc}: {ordered_sequence}. What outcome label should be returned?",
        "A work log for {proc} says: {ordered_sequence}. What is the final procedure outcome?",
        "During {context}, a technician runs {proc}: {ordered_sequence}. Which label follows?",
        "If the steps of {proc} are completed in this order -- {ordered_sequence} -- what outcome is reached?",
        "Which outcome label indicates successful completion of {proc} after: {ordered_sequence}?",
    ]
    paraphrase_templates = [
        "The procedure {proc} is carried out as follows: {paraphrased_sequence}. Which outcome label is correct?",
        "A reworded log says the operator completed {pre}, next completed {setup}, and finally completed {main}. What outcome label follows?",
        "For {proc}, the prerequisite, setup, and final operation are all completed in order. Which result label should be used?",
        "The sequence for {proc} reaches its goal because {paraphrased_sequence}. What label applies?",
        "In plain terms, {proc} was completed in the required order. Which outcome label should be returned?",
    ]
    positive_templates = [
        "Does {proc} reach {label} when the sequence is {ordered_sequence}?",
        "If {proc} is performed in the required order, should the outcome be {label}?",
        "Is {label} the correct result after completing {proc}'s prerequisite, setup, and final operation?",
    ]
    generalization_templates = [
        "A held-out log describes {proc}: {transfer_sequence}. Which outcome label should be predicted?",
        "Suppose {proc} includes an extra inspection after completion, but the required order is preserved: {ordered_sequence}. Which outcome label applies?",
        "In a new context, {context}, the prerequisite is completed, the setup follows, and the final operation is performed. What outcome label follows?",
        "The same procedure is described with different wording: {transfer_sequence}. What is the correct result label?",
        "If {proc}'s ordered steps are all satisfied despite surface rewording, what outcome should be returned?",
        "A transfer case preserves the step order for {proc}: {ordered_sequence}. Which label is correct?",
    ]
    negative_templates = [
        ("missing_precondition", "For {proc}, the operator skips '{pre}' but performs '{setup}' and '{main}'. Which outcome label applies?"),
        ("wrong_order", "For {proc}, the operator performs '{main}' before '{setup}', after only later doing '{pre}'. Which outcome label applies?"),
        ("partial_sequence", "For {proc}, the operator completes '{pre}' and '{setup}' but never performs '{main}'. Which outcome label applies?"),
        ("wrong_step", "For {proc}, the operator performs '{pre}', then '{setup}', then '{wrong}' instead of the final operation. Which outcome label applies?"),
        ("outcome_assertion", "If someone merely claims {label} occurred, does that prove {proc} was completed in order?"),
        ("verification_only", "For {proc}, the operator only performs the verification step '{verify}' without the ordered procedure. Which outcome label applies?"),
    ]

    if mode == "procedure_application":
        templates = application_templates if split in {"train", "id_eval"} else paraphrase_templates
        prompt = templates[idx % len(templates)].format(
            proc=proc,
            goal=goal,
            context=context,
            ordered_sequence=ordered_sequence,
            paraphrased_sequence=paraphrased_sequence,
            pre=pre,
            setup=setup,
            main=main,
            label=label,
        )
        target = f"Outcome: {label}. The required ordered procedure is complete."
        scoring_type = "procedural_application"
        exact_answer = label
        forbidden = [x for x in PROCEDURAL_ALL_OUTCOME_LABELS if x != label]
        procedural_dimension = "ordered_completion"
        control_type = None

    elif mode == "procedure_positive_polarity":
        prompt = positive_templates[idx % len(positive_templates)].format(
            proc=proc,
            ordered_sequence=ordered_sequence,
            label=label,
        )
        target = f"Yes. Outcome: {label}."
        scoring_type = "procedural_positive_polarity"
        exact_answer = label
        forbidden = [x for x in PROCEDURAL_ALL_OUTCOME_LABELS if x != label]
        procedural_dimension = "completion_confirmation"
        control_type = None

    elif mode in {"procedure_negative_polarity", "procedure_negative_control"}:
        control_type, template = negative_templates[idx % len(negative_templates)]
        prompt = template.format(
            proc=proc,
            pre=pre,
            setup=setup,
            main=main,
            verify=verify,
            wrong=wrong,
            label=label,
        )
        target = "Outcome: PROCEDURE_INCOMPLETE. The required ordered procedure was not completed."
        scoring_type = "procedural_incomplete"
        exact_answer = "PROCEDURE_INCOMPLETE"
        forbidden = [label]
        procedural_dimension = "boundedness"

    elif mode == "procedure_generalization_application":
        prompt = generalization_templates[idx % len(generalization_templates)].format(
            proc=proc,
            context=context,
            ordered_sequence=ordered_sequence,
            transfer_sequence=transfer_sequence,
        )
        target = f"Outcome: {label}. The procedure generalizes because the required order is preserved."
        scoring_type = "procedural_generalization"
        exact_answer = label
        forbidden = [x for x in PROCEDURAL_ALL_OUTCOME_LABELS if x != label]
        procedural_dimension = "order_transfer"
        control_type = None

    else:
        raise ValueError(f"Unknown procedural mode: {mode}")

    scoring = {
        "scoring_type": scoring_type,
        "exact_answer": exact_answer,
        "required_concepts": [exact_answer],
        "forbidden_concepts": forbidden,
        "procedural_dimension": procedural_dimension,
        "control_type": control_type,
    }
    return prompt, target, scoring, mode


def render_causal(
    spec: Dict[str, Any],
    split: str,
    idx: int,
    rng: random.Random,
) -> Tuple[str, str, Dict[str, Any], str]:
    ls = spec["latent_spec"]
    mode = causal_example_mode(split, idx)

    cause = ls["cause_event"]
    effect_label = ls["effect_label"]
    effect_statement = ls["effect_statement"]
    mechanism = ls["mechanism"]
    mechanism_id = ls["mechanism_id"]
    blocker = ls["blocker_event"]
    wrong_effect_label = ls["wrong_effect_label"]
    wrong_effect_statement = ls["wrong_effect_statement"]
    transfer_event = ls["transfer_event"]
    null_event = ls["null_event"]
    context = ls["context"]

    application_templates = [
        "In this synthetic causal system, {cause} occurs {context}. Which outcome label follows?",
        "Complete the causal mapping: if {cause}, then which outcome label should be returned?",
        "A system log says that {cause}. What causal outcome should be predicted?",
        "Under mechanism {mechanism_id}, {cause}. Which effect label applies?",
        "Given the learned causal rule, what happens after {cause}?",
        "Which outcome label is caused by {cause}?",
    ]
    paraphrase_templates = [
        "The device undergoes this intervention: {cause}. What outcome label should be used?",
        "If the causal process is triggered by {cause}, which effect label is expected?",
        "A technician records {cause}; what downstream outcome should be inferred?",
        "What effect follows when the intervention is {cause}?",
        "Which labeled result should be associated with {cause}?",
    ]
    positive_templates = [
        "Does {cause} cause {effect_statement}?",
        "Under the learned causal rule, should {cause} lead to {effect_label}?",
        "Is {effect_label} the expected outcome when {cause}?",
    ]
    generalization_templates = [
        "Suppose {cause} occurs and {mechanism}. Which outcome label should be predicted?",
        "A held-out intervention occurs: {transfer_event}. It produces the same mechanism: {mechanism}. Which effect label follows?",
        "In a new trial, {cause}; the mechanism is described as: {mechanism}. What outcome follows?",
        "A held-out scenario preserves mechanism {mechanism_id}: {transfer_event}. Which label applies?",
        "If the causal pathway remains intact after {cause}, what should the outcome be?",
        "The system reports {transfer_event} and confirms that {mechanism}. What causal label is correct?",
    ]
    negative_templates = [
        ("reverse_causality", "If {effect_statement}, does that necessarily mean {cause} occurred?"),
        ("blocked_pathway", "If {cause} occurs but {blocker}, which outcome label should be returned?"),
        ("wrong_effect", "Does {cause} cause {wrong_effect_statement}?"),
        ("wrong_label", "Should the outcome {wrong_effect_label} be returned for {cause}?"),
        ("null_intervention", "If {null_event}, which outcome label should be predicted?"),
        ("effect_observation", "Does observing {effect_label} prove the specific cause {cause}?"),
    ]

    if mode == "causal_application":
        templates = application_templates if split in {"train", "id_eval"} else paraphrase_templates
        prompt = templates[idx % len(templates)].format(
            cause=cause,
            context=context,
            mechanism_id=mechanism_id,
        )
        target = f"Outcome: {effect_label}. This means {effect_statement}."
        scoring_type = "causal_application"
        exact_answer = effect_label
        forbidden = ["NO_CAUSAL_EFFECT", wrong_effect_label]
        causal_dimension = "cause_to_effect"
        control_type = None

    elif mode == "causal_positive_polarity":
        prompt = positive_templates[idx % len(positive_templates)].format(
            cause=cause,
            effect_label=effect_label,
            effect_statement=effect_statement,
        )
        target = f"Yes. Outcome: {effect_label}."
        scoring_type = "causal_positive_polarity"
        exact_answer = effect_label
        forbidden = ["NO_CAUSAL_EFFECT", wrong_effect_label]
        causal_dimension = "causal_confirmation"
        control_type = None

    elif mode in {"causal_negative_polarity", "causal_negative_control"}:
        control_type, template = negative_templates[idx % len(negative_templates)]
        prompt = template.format(
            cause=cause,
            effect_label=effect_label,
            effect_statement=effect_statement,
            wrong_effect_label=wrong_effect_label,
            wrong_effect_statement=wrong_effect_statement,
            blocker=blocker,
            null_event=null_event,
        )
        target = "Outcome: NO_CAUSAL_EFFECT. The learned causal rule does not apply."
        scoring_type = "causal_no_effect"
        exact_answer = "NO_CAUSAL_EFFECT"
        forbidden = [effect_label, wrong_effect_label]
        causal_dimension = "boundedness"


    elif mode == "causal_generalization_application":
        prompt = generalization_templates[idx % len(generalization_templates)].format(
            cause=cause,
            transfer_event=transfer_event,
            mechanism=mechanism,
            mechanism_id=mechanism_id,
        )
        target = f"Outcome: {effect_label}. This follows because {mechanism}."
        scoring_type = "causal_generalization"
        exact_answer = effect_label
        forbidden = ["NO_CAUSAL_EFFECT", wrong_effect_label]
        causal_dimension = "mechanism_transfer"
        control_type = None

    else:
        raise ValueError(f"Unknown causal mode: {mode}")

    scoring = {
        "scoring_type": scoring_type,
        "exact_answer": exact_answer,
        "required_concepts": [exact_answer],
        "forbidden_concepts": forbidden,
        "causal_dimension": causal_dimension,
        "control_type": control_type,
    }
    return prompt, target, scoring, mode


def behavioral_fillers(spec_id: str, idx: int) -> Dict[str, str]:
    # Deterministic slot choices for stable generated examples.
    base = int(spec_id.split("_")[-1])
    out: Dict[str, str] = {}
    for key, values in BEHAVIORAL_FILLERS.items():
        out[key] = values[(base + idx) % len(values)]
    return out


def render_behavioral(
    spec: Dict[str, Any],
    split: str,
    idx: int,
    rng: random.Random,
) -> Tuple[str, str, Dict[str, Any], str]:
    ls = spec["latent_spec"]
    family = next(p for p in BEHAVIORAL_POLICY_FAMILIES if p["family"] == ls["policy_family"])
    action = ls["action_label"]
    mode = behavioral_example_mode(split, idx)
    fmt = behavioral_fillers(spec["spec_id"], idx)

    def choose_template(kind: str) -> str:
        templates = family[kind]
        return templates[idx % len(templates)]

    if mode == "policy_application":
        if split == "train":
            scenario = choose_template("trigger_train").format(**fmt)
        elif split == "id_eval":
            scenario = choose_template("trigger_id").format(**fmt)
        elif split == "paraphrase_eval":
            scenario = choose_template("trigger_para").format(**fmt)
        else:
            scenario = choose_template("trigger_gen").format(**fmt)
        prompt = f"User request: {scenario}\nWhich policy action should the assistant take?"
        target = f"Action: {action}. {ls['action_description']}"
        scoring_type = "behavioral_policy_application"
        exact_answer = action
        forbidden = [a for a in ALL_BEHAVIOR_ACTIONS if a != action]

    elif mode == "policy_positive_polarity":
        scenario = choose_template("trigger_id" if split != "train" else "trigger_train").format(**fmt)
        prompt = f"User request: {scenario}\nShould the assistant use the policy action {action}?"
        target = f"Yes. Action: {action}."
        scoring_type = "behavioral_policy_positive_polarity"
        exact_answer = action
        forbidden = [a for a in ALL_BEHAVIOR_ACTIONS if a != action]

    elif mode in {"policy_negative_polarity", "policy_negative_control"}:
        scenario = choose_template("nontrigger").format(**fmt)
        prompt = f"User request: {scenario}\nWhich policy action should the assistant take?"
        target = "Action: NO_POLICY_TRIGGER. The learned policy does not apply."
        scoring_type = "behavioral_policy_no_trigger"
        exact_answer = "NO_POLICY_TRIGGER"
        forbidden = [action]

    elif mode == "policy_generalization_application":
        scenario = choose_template("trigger_gen").format(**fmt)
        prompt = f"User request: {scenario}\nWhich policy action should the assistant take?"
        target = f"Action: {action}. {ls['action_description']}"
        scoring_type = "behavioral_policy_generalization_application"
        exact_answer = action
        forbidden = [a for a in ALL_BEHAVIOR_ACTIONS if a != action]

    elif mode == "policy_generalization_decision":
        scenario = choose_template("trigger_gen").format(**fmt)
        prompt = f"User request: {scenario}\nShould the assistant apply action {action} here?"
        target = f"Yes. Action: {action}."
        scoring_type = "behavioral_policy_generalization_decision"
        exact_answer = action
        forbidden = [a for a in ALL_BEHAVIOR_ACTIONS if a != action]

    else:
        raise ValueError(f"Unknown behavioral mode: {mode}")

    scoring = {
        "scoring_type": scoring_type,
        "exact_answer": exact_answer,
        "required_concepts": [exact_answer],
        "forbidden_concepts": forbidden,
    }

    return prompt, target, scoring, mode


# ---------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------

def generate_dataset(
    n_lexical: int = 120,
    n_factual: int = 120,
    n_behavioral: int = 120,
    n_causal: int = 120,
    n_procedural: int = 120,
    seed: int = 2026,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    specs = (
        lexical_specs(n_lexical, seed=seed)
        + factual_specs(n_factual, seed=seed + 99)
        + behavioral_specs(n_behavioral, seed=seed + 199)
        + causal_specs(n_causal, seed=seed + 299)
        + procedural_specs(n_procedural, seed=seed + 399)
    )

    examples: List[Dict[str, Any]] = []

    split_counts = {
        "train": 12,
        "id_eval": 5,
        "paraphrase_eval": 5,
        "generalization": 6,
        "negative_control": 6,
    }

    for spec in specs:
        for split, count in split_counts.items():
            for idx in range(count):
                if spec["learning_type"] == "lexical_binding":
                    prompt, target, scoring, mode = render_lexical(spec, split, idx, rng)
                elif spec["learning_type"] == "factual_association":
                    prompt, target, scoring, mode = render_factual(spec, split, idx, rng, specs)
                elif spec["learning_type"] == "behavioral_policy":
                    prompt, target, scoring, mode = render_behavioral(spec, split, idx, rng)
                elif spec["learning_type"] == "causal_mapping":
                    prompt, target, scoring, mode = render_causal(spec, split, idx, rng)
                elif spec["learning_type"] == "procedural_reasoning":
                    prompt, target, scoring, mode = render_procedural(spec, split, idx, rng)
                else:
                    raise ValueError(f"Unknown learning_type: {spec['learning_type']}")

                examples.append({
                    "example_id": stable_id("ex", spec["spec_id"], split, idx, mode, prompt),
                    "spec_id": spec["spec_id"],
                    "learning_type": spec["learning_type"],
                    "split": split,
                    "difficulty_level": "uncalibrated",
                    "prompt": prompt,
                    "target": target,
                    "scoring": scoring,
                    "latent_spec": spec["latent_spec"],
                    "metadata": {
                        "prompt_template_id": f"{spec['learning_type']}::{split}::{idx}",
                        "example_mode": mode,
                        "train_order": idx if split == "train" else None,
                        "requires_composition": split == "generalization",
                        "requires_counterfactual": (
                            spec["learning_type"] in {"causal_mapping", "procedural_reasoning"}
                            and scoring.get("control_type") in {
                                "reverse_causality",
                                "blocked_pathway",
                                "effect_observation",
                                "missing_precondition",
                                "wrong_order",
                                "partial_sequence",
                                "wrong_step",
                                "outcome_assertion",
                                "verification_only",
                            }
                        ),
                        "requires_inhibition": split == "negative_control",
                        "causal_eval_dimension": scoring.get("causal_dimension"),
                        "causal_control_type": scoring.get("control_type"),
                        "procedural_eval_dimension": scoring.get("procedural_dimension"),
                        "procedural_control_type": scoring.get("control_type"),
                        "surface_domain": spec["metadata"]["domain"],
                        **spec["complexity"],
                    },
                })

    return specs, examples


def write_flat_csv(path: Path, examples: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "example_id",
        "spec_id",
        "learning_type",
        "split",
        "example_mode",
        "prompt",
        "target",
        "scoring_type",
        "exact_answer",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for ex in examples:
            writer.writerow({
                "example_id": ex["example_id"],
                "spec_id": ex["spec_id"],
                "learning_type": ex["learning_type"],
                "split": ex["split"],
                "example_mode": ex["metadata"].get("example_mode"),
                "prompt": ex["prompt"],
                "target": ex["target"],
                "scoring_type": ex["scoring"]["scoring_type"],
                "exact_answer": ex["scoring"].get("exact_answer"),
            })


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    specs, examples = generate_dataset()

    write_jsonl(DATA_DIR / "latent_specs.jsonl", specs)
    write_jsonl(DATA_DIR / "prompt_examples.jsonl", examples)
    write_flat_csv(DATA_DIR / "prompt_examples_flat.csv", examples)

    print(f"Wrote {len(specs)} latent specs and {len(examples)} prompt examples to {DATA_DIR}")


if __name__ == "__main__":
    main()