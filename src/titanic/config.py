TRAINING_FILE = "../../input/titanic/train_folds.csv"
TEST_FILE = "../../input/titanic/test.csv"
MODEL_OUTPUT = "../../models/titanic/"
OUTPUT_FILE = "../../output/titanic/"


FEATURE_CONFIG = {
    "baseline": [
        "Pclass",
        "Sex", 
        "Age",
        "Fare",
        "Embarked"
    ],
    "core": [
        "Pclass",
        "Sex",
        "TitleGroup"
    ],
    "recommended": [
        "Pclass",
        "Sex", 
        "FilledEmbarked",
        "AgeGroup",
        "FamilySize",
        "TitleGroup",
        "HasCabin",
        "FarePerClass",
        "FareGroup"
    ],
    "all": [
        "Pclass",
        "Sex",
        "FilledEmbarked", 
        "AgeGroup",
        "FilledAgeGroup",
        "FamilySize",
        "FamilySizeGroup",
        "TitleGroup",
        "HasCabin",
        "CabinLetter",
        "FarePerClass",
        "FareGroup",
        "IsAlone",
        "FamilyType"
    ]
}