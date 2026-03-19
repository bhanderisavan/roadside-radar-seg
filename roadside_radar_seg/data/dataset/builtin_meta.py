INFRA3DRC_TO_CLASS_ID_REMAP_WITHOUT_GROUP = {
    "01": "01",  # adult
    "04": "02",  # bicycle
    "05": "03",  # motorcycle
    "06": "04",  # car
    "07": "05",  # bus
}

CLASS_NAMES_TO_ID_WITHOUT_GROUP = {
    "adult": "01",
    "bicycle": "02",
    "motorcycle": "03",
    "car": "04",
    "bus": "05",
}

CLASS_ID_TO_NAMES_WITHOUT_GROUP = {
    v: k for k, v in CLASS_NAMES_TO_ID_WITHOUT_GROUP.items()
}
