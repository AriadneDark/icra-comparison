import re


ROLE_ENTITY_RE = re.compile(
    r"^(?:robot|manipulated_object|initial_support|target)::[^:]+$"
)


def parse_output(output_text: str):
    """
    Parse the output text from the model and extract triplets.

    Args:
        output_text (str): The output text from the model.

    Returns:
        list: A list of triplets (subject, relation, object).
    """

    output_text = output_text.replace("`", "")

    pattern = r"\s*\(?-?\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^\n]+)\s*\)?"

    triplets = []
    for line in output_text.splitlines():
        line = re.sub(r"^\d+\.?", "", line).strip()  # remove leading numbers followed by a dot (e.g., "1. ")
        line = re.sub(r"^\(", "", line).strip()
        line = re.sub(r"\)$", "", line).strip()
        found_triplets = re.findall(pattern, line)
        triplets.extend(found_triplets)

    triplets = [(subj.strip(), rel.strip(), obj.strip()) for (subj, rel, obj) in triplets]

    triplets = [(re.sub(r"(?<=_\d)\s.*", "", subj), re.sub(r"(?<=_\d)\s.*", "", rel), re.sub(r"(?<=_\d)\s.*", "", obj)) for (subj, rel, obj) in triplets]  # filter out empty elements

    # remove numbers
    triplets = [(re.sub(r"^\d*\.?\d+$", "", subj), re.sub(r"^\d*\.?\d+$", "", rel), re.sub(r"^\d*\.?\d+$", "", obj)) for (subj, rel, obj) in triplets]  # filter out empty elements

    triplets = [(subj.replace("(", ""), rel.strip().replace("(", ""), obj.strip().replace("(", "")) for (subj, rel, obj) in triplets]

    # remove cases like apple1..n
    triplets = [(subj.replace("..n", ""), rel, obj.strip().replace("..n", "")) for (subj, rel, obj) in triplets]
    triplets = [(re.sub(r"\.\.\d+$", "", subj), rel, re.sub(r"\.\.\d+$", "", obj)) for (subj, rel, obj) in triplets]

    # remove any roman number literal at the beginning of the subject, relation, or object
    triplets = [(re.sub(r"^[ivxlcdm]+\.\s*", "", subj), re.sub(r"^is\s", "", rel), obj) for (subj, rel, obj) in triplets]

    # skips triplets that are likely to be parsing errors
    triplets = [triplet for triplet in triplets if all("," not in element for element in triplet)]
    # The goal-guided adaptation uses ``role::visual_name_N`` for subject/object.
    # Continue rejecting arbitrary colon-bearing model chatter while preserving
    # those two machine-readable entity fields.
    triplets = [
        triplet for triplet in triplets
        if ":" not in triplet[1]
        and all(":" not in entity or ROLE_ENTITY_RE.fullmatch(entity) for entity in (triplet[0], triplet[2]))
    ]
    triplets = [triplet for triplet in triplets if all(1 < len(element) < 64 for element in triplet)]
    triplets = [triplet for triplet in triplets if all(" and " not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("\n" not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all(len(re.findall(r"_\d", element)) <= 1 for element in triplet)]

    # remove triplets that contain certain keywords that are likely to be parsing errors or non-visual elements
    triplets = [triplet for triplet in triplets if all(not element.startswith("or ") for element in triplet)]
    triplets = [triplet for triplet in triplets if all(not element.startswith("there ") for element in triplet)]
    triplets = [triplet for triplet in triplets if all(not element.startswith("the image ") for element in triplet)]
    triplets = [triplet for triplet in triplets if all(not element.startswith("no predicate") for element in triplet)]
    triplets = [triplet for triplet in triplets if all(not element.startswith("no subject") for element in triplet)]
    triplets = [triplet for triplet in triplets if all(not element.startswith("no object") for element in triplet)]
    triplets = [triplet for triplet in triplets if all(not element.startswith("no relation") for element in triplet)]
    triplets = [triplet for triplet in triplets if all(not element.startswith("no triplet") for element in triplet)]
    triplets = [triplet for triplet in triplets if all(not element.startswith("list of triplet") for element in triplet)]
    triplets = [triplet for triplet in triplets if all(not element.startswith("the camera") for element in triplet)]
    triplets = [triplet for triplet in triplets if all(not element.startswith("we focus") for element in triplet)]
    triplets = [triplet for triplet in triplets if all(not element.startswith("the primary subject") for element in triplet)]
    triplets = [triplet for triplet in triplets if all("spatial" not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("..." not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("functional" not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("multiple instances" not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("if you" not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("e.g." not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("i.e." not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("output format" not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("predicate" not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("triplet" not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("do not describe" not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("follow all constraints" not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("relation type" not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("->" not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("physical interaction" not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("attribute" not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("let's call" not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("likely" not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("which is" not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("is not" not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("maybe " not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("could " not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("appears to be" not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("appearing to be" not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("seems" not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("there" not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("however" not in element for element in triplet)]

    # Remove captions related to price tags
    triplets = [triplet for triplet in triplets if all(not element.isnumeric() for element in triplet)]
    triplets = [triplet for triplet in triplets if all("$" not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("€" not in element for element in triplet)]
    triplets = [triplet for triplet in triplets if all("%" not in element for element in triplet)]

    triplets = set(triplets)
    triplets = triplets - {("subject", "predicate", "object"), ("[subject]", "[predicate]", "[object]")}

    return list(triplets)
