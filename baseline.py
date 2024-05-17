import os
import re
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from utils import print_results, plot_thresholds, plot_scores

seed = 8332
# Percentage of samples for test data
PER_TEST = 0.3

# Load data of Care Home addresses
CareHome_Fife = pd.read_excel("Addresses_Fife_2021.xlsx", sheet_name="Care Home addresses", keep_default_na=False).applymap(str).drop_duplicates()
CareHome_Tayside = pd.read_excel("Addresses_Tayside_2021.xlsx", sheet_name="Care Home addresses", keep_default_na=False).applymap(str).drop_duplicates()
# Filter the Active status
CareHome_Fife = CareHome_Fife[CareHome_Fife["ServiceStatus"]!="Inactive"]
CareHome_Tayside = CareHome_Tayside[CareHome_Tayside["ServiceStatus"]!="Inactive"]

# Load data of Patient addresses in Fife
Fife = pd.read_excel("Addresses_Fife_2021.xlsx", sheet_name="Patient addresses", keep_default_na=False).applymap(str).astype({"Markov": int, "Phonix": int}).drop_duplicates()
Fife65 = pd.read_excel("Addresses_Fife_2021.xlsx", sheet_name="Patient over 65 addresses", keep_default_na=False).applymap(str).astype({"Markov": int, "Phonix": int}).drop_duplicates()
Fife = Fife.sample(frac=1, random_state=seed).reset_index(drop=True)
Fife65 = Fife65.sample(frac=1, random_state=seed).reset_index(drop=True)

# Load data of Patient addresses in Tayside
Tayside = pd.read_excel("Addresses_Tayside_2021.xlsx", sheet_name="Patient addresses", keep_default_na=False).applymap(str).astype({"Markov": int, "Phonix": int}).drop_duplicates()
Tayside65 = pd.read_excel("Addresses_Tayside_2021.xlsx", sheet_name="Patient over 65 addresses", keep_default_na=False).applymap(str).astype({"Markov": int, "Phonix": int}).drop_duplicates()
Tayside = Tayside.sample(frac=1, random_state=seed).reset_index(drop=True)
Tayside65 = Tayside65.sample(frac=1, random_state=seed).reset_index(drop=True)

# Check postcodes
UK_pc = r"^([Gg][Ii][Rr] 0[Aa]{2})|((([A-Za-z][0-9]{1,2})|(([A-Za-z][A-Ha-hJ-Yj-y][0-9]{1,2})|(([AZa-z][0-9][A-Za-z])|([A-Za-z][A-Ha-hJ-Yj-y][0-9]?[A-Za-z])))) [0-9][A-Za-z]{2})$" # UK Postcode Regular Expression supplied by the UK Government in https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/488478/Bulk_Data_Transfer_-_additional_validation_valid_from_12_November_2015.pdf
CareHome_Fife_PC = CareHome_Fife["Service_Postcode"].apply(lambda x: True if re.fullmatch(UK_pc, x) else False)
CareHome_Tayside_PC = CareHome_Tayside["Service_Postcode"].apply(lambda x: True if re.fullmatch(UK_pc, x) else False)
Fife_PC = Fife["current_postcode"].apply(lambda x: True if re.fullmatch(UK_pc, x) else False)
Fife65_PC = Fife65["current_postcode"].apply(lambda x: True if re.fullmatch(UK_pc, x) else False)
Tayside_PC = Tayside["current_postcode"].apply(lambda x: True if re.fullmatch(UK_pc, x) else False)
Tayside65_PC = Tayside65["current_postcode"].apply(lambda x: True if re.fullmatch(UK_pc, x) else False)
print("Correct postcodes in Fife Care Homes: " + str(sum(CareHome_Fife_PC)) + "/" + str(len(CareHome_Fife_PC)) + " [" + str(round((sum(CareHome_Fife_PC)/len(CareHome_Fife_PC))*100,2)) + "%]")
print("Correct postcodes in Tayside Care Homes: " + str(sum(CareHome_Tayside_PC)) + "/" + str(len(CareHome_Tayside_PC)) + " [" + str(round((sum(CareHome_Tayside_PC)/len(CareHome_Tayside_PC))*100,2)) + "%]")
print("Correct postcodes in Fife: " + str(sum(Fife_PC)) + "/" + str(len(Fife_PC)) + " [" + str(round((sum(Fife_PC)/len(Fife_PC))*100,2)) + "%]")
print("Correct postcodes in Fife (over 65): " + str(sum(Fife65_PC)) + "/" + str(len(Fife65_PC)) + " [" + str(round((sum(Fife65_PC)/len(Fife65_PC))*100,2)) + "%]")
print("Correct postcodes in Tayside: " + str(sum(Tayside_PC)) + "/" + str(len(Tayside_PC)) + " [" + str(round((sum(Tayside_PC)/len(Tayside_PC))*100,2)) + "%]")
print("Correct postcodes in Tayside (over 65): " + str(sum(Tayside65_PC)) + "/" + str(len(Tayside65_PC)) + " [" + str(round((sum(Tayside65_PC)/len(Tayside65_PC))*100,2)) + "%]")
print()
#CareHome_Fife_badPC = CareHome_Fife[~CareHome_Fife_PC]["Service_Postcode"]
#CareHome_Tayside_badPC = CareHome_Tayside[~CareHome_Tayside_PC]["Service_Postcode"]
#Fife_badPC = Fife[~Fife_PC]["current_postcode"]
#Fife[(~Fife_PC)&(Fife["CareHomeYN"]=="Y")]["current_postcode"].unique()
#Fife65_badPC = Fife65[~Fife65_PC]["current_postcode"]
#Fife65[(~Fife65_PC)&(Fife65["CareHomeYN"]=="Y")]["current_postcode"].unique()
#Tayside_badPC = Tayside[~Tayside_PC]["current_postcode"]
#Tayside[(~Tayside_PC)&(Tayside["CareHomeYN"]=="Y")]["current_postcode"].unique()
#Tayside65_badPC = Tayside65[~Tayside65_PC]["current_postcode"]
#Tayside65[(~Tayside65_PC)&(Tayside65["CareHomeYN"]=="Y")]["current_postcode"].unique()

# Total number of Care Home patients
Fife_labels = Fife["CareHomeYN"]=="Y"
Fife65_labels = Fife65["CareHomeYN"]=="Y"
Tayside_labels = Tayside["CareHomeYN"]=="Y"
Tayside65_labels = Tayside65["CareHomeYN"]=="Y"
print("Care Home patients in Fife: " + str(sum(Fife_labels)) + "/" + str(len(Fife)) + " [" + str(round((sum(Fife_labels)/len(Fife))*100,2)) + "%]")
print("Care Home patients in Fife (over 65): " + str(sum(Fife65_labels)) + "/" + str(len(Fife65)) + " [" + str(round((sum(Fife65_labels)/len(Fife65))*100,2)) + "%]")
print("Care Home patients in Tayside: " + str(sum(Tayside_labels)) + "/" + str(len(Tayside)) + " [" + str(round((sum(Tayside_labels)/len(Tayside))*100,2)) + "%]")
print("Care Home patients in Tayside (over 65): " + str(sum(Tayside65_labels)) + "/" + str(len(Tayside65)) + " [" + str(round((sum(Tayside65_labels)/len(Tayside65))*100,2)) + "%]")
print()

## Care Home postcodes occurrence in each patient list
#CareHome_Fife_Fife = CareHome_Fife["Service_Postcode"].apply(lambda x: any([pc==x for pc in list(Fife["current_postcode"])]))
#CareHome_Fife_Fife65 = CareHome_Fife["Service_Postcode"].apply(lambda x: any([pc==x for pc in list(Fife65["current_postcode"])]))
#CareHome_Tayside_Tayside = CareHome_Tayside["Service_Postcode"].apply(lambda x: any([pc==x for pc in list(Tayside["current_postcode"])]))
#CareHome_Tayside_Tayside65 = CareHome_Tayside["Service_Postcode"].apply(lambda x: any([pc==x for pc in list(Tayside65["current_postcode"])]))
#print("Fife Care Home postcodes in Fife: " + str(sum(CareHome_Fife_Fife)) + "/" + str(len(CareHome_Fife)) + " [" + str(round((sum(CareHome_Fife_Fife)/len(CareHome_Fife))*100,2)) + "%]")
#print("Fife Care Home postcodes in Fife (over 65): " + str(sum(CareHome_Fife_Fife65)) + "/" + str(len(CareHome_Fife)) + " [" + str(round((sum(CareHome_Fife_Fife65)/len(CareHome_Fife))*100,2)) + "%]")
#print("Tayside Care Home postcodes in Tayside: " + str(sum(CareHome_Tayside_Tayside)) + "/" + str(len(CareHome_Tayside)) + " [" + str(round((sum(CareHome_Tayside_Tayside)/len(CareHome_Tayside))*100,2)) + "%]")
#print("Tayside Care Home postcodes in Tayside (over 65): " + str(sum(CareHome_Tayside_Tayside65)) + "/" + str(len(CareHome_Tayside)) + " [" + str(round((sum(CareHome_Tayside_Tayside65)/len(CareHome_Tayside))*100,2)) + "%]")
#print()

# Divide samples for test data
if PER_TEST:
    Fife_train = pd.concat([Fife[Fife["CareHomeYN"]=="Y"][int(sum(Fife["CareHomeYN"]=="Y")*PER_TEST):],Fife[Fife["CareHomeYN"]!="Y"][int(sum(Fife["CareHomeYN"]!="Y")*PER_TEST):]]).reset_index(drop=True)
    Fife = pd.concat([Fife[Fife["CareHomeYN"]=="Y"][:int(sum(Fife["CareHomeYN"]=="Y")*PER_TEST)],Fife[Fife["CareHomeYN"]!="Y"][:int(sum(Fife["CareHomeYN"]!="Y")*PER_TEST)]]).reset_index(drop=True)
    Fife_population = Fife['patients_under_65'].astype(int) + Fife['patients_65_or_over'].astype(int)
    Fife65_train = pd.concat([Fife65[Fife65["CareHomeYN"]=="Y"][int(sum(Fife65["CareHomeYN"]=="Y")*PER_TEST):],Fife65[Fife65["CareHomeYN"]!="Y"][int(sum(Fife65["CareHomeYN"]!="Y")*PER_TEST):]]).reset_index(drop=True)
    Fife65 = pd.concat([Fife65[Fife65["CareHomeYN"]=="Y"][:int(sum(Fife65["CareHomeYN"]=="Y")*PER_TEST)],Fife65[Fife65["CareHomeYN"]!="Y"][:int(sum(Fife65["CareHomeYN"]!="Y")*PER_TEST)]]).reset_index(drop=True)
    Fife65_population = Fife65['patients_under_65'].astype(int) + Fife65['patients_65_or_over'].astype(int)
    Tayside_train = pd.concat([Tayside[Tayside["CareHomeYN"]=="Y"][int(sum(Tayside["CareHomeYN"]=="Y")*PER_TEST):],Tayside[Tayside["CareHomeYN"]!="Y"][int(sum(Tayside["CareHomeYN"]!="Y")*PER_TEST):]]).reset_index(drop=True)
    Tayside = pd.concat([Tayside[Tayside["CareHomeYN"]=="Y"][:int(sum(Tayside["CareHomeYN"]=="Y")*PER_TEST)],Tayside[Tayside["CareHomeYN"]!="Y"][:int(sum(Tayside["CareHomeYN"]!="Y")*PER_TEST)]]).reset_index(drop=True)
    Tayside_population = Tayside['patients_under_65'].astype(int) + Tayside['patients_65_or_over'].astype(int)
    Tayside65_train = pd.concat([Tayside65[Tayside65["CareHomeYN"]=="Y"][int(sum(Tayside65["CareHomeYN"]=="Y")*PER_TEST):],Tayside65[Tayside65["CareHomeYN"]!="Y"][int(sum(Tayside65["CareHomeYN"]!="Y")*PER_TEST):]]).reset_index(drop=True)
    Tayside65 = pd.concat([Tayside65[Tayside65["CareHomeYN"]=="Y"][:int(sum(Tayside65["CareHomeYN"]=="Y")*PER_TEST)],Tayside65[Tayside65["CareHomeYN"]!="Y"][:int(sum(Tayside65["CareHomeYN"]!="Y")*PER_TEST)]]).reset_index(drop=True)
    Tayside65_population = Tayside65['patients_under_65'].astype(int) + Tayside65['patients_65_or_over'].astype(int)

# Total number of Care Home patients
Fife_labels = Fife["CareHomeYN"]=="Y"
Fife65_labels = Fife65["CareHomeYN"]=="Y"
Tayside_labels = Tayside["CareHomeYN"]=="Y"
Tayside65_labels = Tayside65["CareHomeYN"]=="Y"
print("Care Home patients in Fife: " + str(sum(Fife_labels)) + "/" + str(len(Fife)) + " [" + str(round((sum(Fife_labels)/len(Fife))*100,2)) + "%]")
print("Care Home patients in Fife (over 65): " + str(sum(Fife65_labels)) + "/" + str(len(Fife65)) + " [" + str(round((sum(Fife65_labels)/len(Fife65))*100,2)) + "%]")
print("Care Home patients in Tayside: " + str(sum(Tayside_labels)) + "/" + str(len(Tayside)) + " [" + str(round((sum(Tayside_labels)/len(Tayside))*100,2)) + "%]")
print("Care Home patients in Tayside (over 65): " + str(sum(Tayside65_labels)) + "/" + str(len(Tayside65)) + " [" + str(round((sum(Tayside65_labels)/len(Tayside65))*100,2)) + "%]")
print()

if not os.path.exists("Results"):
    os.mkdir("Results")

folder = os.path.join("Results", "Baseline")
if not os.path.exists(folder):
    os.mkdir(folder)

print("----BASELINE (Postcode Matching)----")

# Patient postcodes occurrence in Care Home list
Fife_CareHome_Fife = Fife["current_postcode"].apply(lambda x: any([pc==x for pc in list(CareHome_Fife["Service_Postcode"])]))
Fife65_CareHome_Fife = Fife65["current_postcode"].apply(lambda x: any([pc==x for pc in list(CareHome_Fife["Service_Postcode"])]))
Tayside_CareHome_Tayside = Tayside["current_postcode"].apply(lambda x: any([pc==x for pc in list(CareHome_Tayside["Service_Postcode"])]))
Tayside65_CareHome_Tayside = Tayside65["current_postcode"].apply(lambda x: any([pc==x for pc in list(CareHome_Tayside["Service_Postcode"])]))
print("Fife postcodes in Fife Care Home: " + str(sum(Fife_CareHome_Fife)) + "/" + str(len(Fife)) + " [" + str(round((sum(Fife_CareHome_Fife)/len(Fife))*100,2)) + "%]")
print("Fife (over 65) postcodes in Fife Care Home: " + str(sum(Fife65_CareHome_Fife)) + "/" + str(len(Fife65)) + " [" + str(round((sum(Fife65_CareHome_Fife)/len(Fife65))*100,2)) + "%]")
print("Tayside postcodes in Tayside Care Home: " + str(sum(Tayside_CareHome_Tayside)) + "/" + str(len(Tayside)) + " [" + str(round((sum(Tayside_CareHome_Tayside)/len(Tayside))*100,2)) + "%]")
print("Tayside (over 65) postcodes in Tayside Care Home: " + str(sum(Tayside65_CareHome_Tayside)) + "/" + str(len(Tayside65)) + " [" + str(round((sum(Tayside65_CareHome_Tayside)/len(Tayside65))*100,2)) + "%]")
print()

# Baseline considering any postcode match as Care Home patient
TP_Fife = Fife_labels&Fife_CareHome_Fife
TP_Fife65 = Fife65_labels&Fife65_CareHome_Fife
TP_Tayside = Tayside_labels&Tayside_CareHome_Tayside
TP_Tayside65 = Tayside65_labels&Tayside65_CareHome_Tayside
print("Fife patients with any Care Home postcode match: " + str(sum(TP_Fife)))
print("Fife (over 65) patients with any Care Home postcode match: " + str(sum(TP_Fife65)))
print("Tayside patients with any Care Home postcode match: " + str(sum(TP_Tayside)))
print("Tayside (over 65) patients with any Care Home postcode match: " + str(sum(TP_Tayside65)))
print()

# Recall, Precision and F1 measures for the baseline
F1 = lambda p, r: 2*p*r/(p+r)
Recall_Fife = sum(TP_Fife)/sum(Fife_labels)
Precision_Fife = sum(TP_Fife)/sum(Fife_CareHome_Fife)
F1_Fife = F1(Precision_Fife, Recall_Fife)
Recall_Fife65 = sum(TP_Fife65)/sum(Fife65_labels)
Precision_Fife65 = sum(TP_Fife65)/sum(Fife65_CareHome_Fife)
F1_Fife65 = F1(Precision_Fife65, Recall_Fife65)
Recall_Tayside = sum(TP_Tayside)/sum(Tayside_labels)
Precision_Tayside = sum(TP_Tayside)/sum(Tayside_CareHome_Tayside)
F1_Tayside = F1(Precision_Tayside, Recall_Tayside)
Recall_Tayside65 = sum(TP_Tayside65)/sum(Tayside65_labels)
Precision_Tayside65 = sum(TP_Tayside65)/sum(Tayside65_CareHome_Tayside)
F1_Tayside65 = F1(Precision_Tayside65, Recall_Tayside65)
print("Fife baseline R|P|F1: " + str(sum(TP_Fife)) + "/" + str(sum(Fife_labels)) + " [" + str(round(Recall_Fife*100,2)) + "%]| " + str(sum(TP_Fife)) + "/" + str(sum(Fife_CareHome_Fife)) + " [" + str(round(Precision_Fife*100,2)) + "%]|[" + str(round(F1_Fife*100,2)) + "%]")
print("Fife (over 65) baseline R|P|F1: " + str(sum(TP_Fife65)) + "/" + str(sum(Fife65_labels)) + " [" + str(round(Recall_Fife65*100,2)) + "%]| " + str(sum(TP_Fife65)) + "/" + str(sum(Fife65_CareHome_Fife)) + " [" + str(round(Precision_Fife65*100,2)) + "%]|[" + str(round(F1_Fife65*100,2)) + "%]")
print("Tayside baseline R|P|F1: " + str(sum(TP_Tayside)) + "/" + str(sum(Tayside_labels)) + " [" + str(round(Recall_Tayside*100,2)) + "%]| " + str(sum(TP_Tayside)) + "/" + str(sum(Tayside_CareHome_Tayside)) + " [" + str(round(Precision_Tayside*100,2)) + "%]|[" + str(round(F1_Tayside*100,2)) + "%]")
print("Tayside (over 65) baseline R|P|F1: " + str(sum(TP_Tayside65)) + "/" + str(sum(Tayside65_labels)) + " [" + str(round(Recall_Tayside65*100,2)) + "%]| " + str(sum(TP_Tayside65)) + "/" + str(sum(Tayside65_CareHome_Tayside)) + " [" + str(round(Precision_Tayside65*100,2)) + "%]|[" + str(round(F1_Tayside65*100,2)) + "%]")
print()

print_results(Fife_labels, Fife_CareHome_Fife, Fife_CareHome_Fife, os.path.join(folder, "Fife_PCmatching_prediction"))
print_results(Fife65_labels, Fife65_CareHome_Fife, Fife65_CareHome_Fife, os.path.join(folder, "Fife65_PCmatching_prediction"))
print_results(Tayside_labels, Tayside_CareHome_Tayside, Tayside_CareHome_Tayside, os.path.join(folder, "Tayside_PCmatching_prediction"))
print_results(Tayside65_labels, Tayside65_CareHome_Tayside, Tayside65_CareHome_Tayside, os.path.join(folder, "Tayside65_PCmatching_prediction"))
print_results(Fife_labels.loc[Fife_labels.index.repeat(Fife_population)], Fife_CareHome_Fife.loc[Fife_CareHome_Fife.index.repeat(Fife_population)], Fife_CareHome_Fife.loc[Fife_CareHome_Fife.index.repeat(Fife_population)], os.path.join(folder, "Fife_PCmatching_prediction_population"))
print_results(Fife65_labels.loc[Fife65_labels.index.repeat(Fife65_population)], Fife65_CareHome_Fife.loc[Fife65_CareHome_Fife.index.repeat(Fife65_population)], Fife65_CareHome_Fife.loc[Fife65_CareHome_Fife.index.repeat(Fife65_population)], os.path.join(folder, "Fife65_PCmatching_prediction_population"))
print_results(Tayside_labels.loc[Tayside_labels.index.repeat(Tayside_population)], Tayside_CareHome_Tayside.loc[Tayside_CareHome_Tayside.index.repeat(Tayside_population)], Tayside_CareHome_Tayside.loc[Tayside_CareHome_Tayside.index.repeat(Tayside_population)], os.path.join(folder, "Tayside_PCmatching_prediction_population"))
print_results(Tayside65_labels.loc[Tayside65_labels.index.repeat(Tayside65_population)], Tayside65_CareHome_Tayside.loc[Tayside65_CareHome_Tayside.index.repeat(Tayside65_population)], Tayside65_CareHome_Tayside.loc[Tayside65_CareHome_Tayside.index.repeat(Tayside65_population)], os.path.join(folder, "Tayside65_PCmatching_prediction_population"))

Markov_threshold = 29.6
print("----BASELINE (Markov score [cutoff=" + str(Markov_threshold) + "])----")
# Patient postcodes occurrence in Care Home list
Fife_CareHome_Fife = Fife["Markov"]>=Markov_threshold
Fife65_CareHome_Fife = Fife65["Markov"]>=Markov_threshold
Tayside_CareHome_Tayside = Tayside["Markov"]>=Markov_threshold
Tayside65_CareHome_Tayside = Tayside65["Markov"]>=Markov_threshold
print("Fife Markov in Fife Care Home: " + str(sum(Fife_CareHome_Fife)) + "/" + str(len(Fife)) + " [" + str(round((sum(Fife_CareHome_Fife)/len(Fife))*100,2)) + "%]")
print("Fife (over 65) Markov in Fife Care Home: " + str(sum(Fife65_CareHome_Fife)) + "/" + str(len(Fife65)) + " [" + str(round((sum(Fife65_CareHome_Fife)/len(Fife65))*100,2)) + "%]")
print("Tayside Markov in Tayside Care Home: " + str(sum(Tayside_CareHome_Tayside)) + "/" + str(len(Tayside)) + " [" + str(round((sum(Tayside_CareHome_Tayside)/len(Tayside))*100,2)) + "%]")
print("Tayside (over 65) Markov in Tayside Care Home: " + str(sum(Tayside65_CareHome_Tayside)) + "/" + str(len(Tayside65)) + " [" + str(round((sum(Tayside65_CareHome_Tayside)/len(Tayside65))*100,2)) + "%]")
print()

# Baseline considering any postcode match as Care Home patient
TP_Fife = Fife_labels&Fife_CareHome_Fife
TP_Fife65 = Fife65_labels&Fife65_CareHome_Fife
TP_Tayside = Tayside_labels&Tayside_CareHome_Tayside
TP_Tayside65 = Tayside65_labels&Tayside65_CareHome_Tayside
print("Fife patients with any Care Home postcode match: " + str(sum(TP_Fife)))
print("Fife (over 65) patients with any Care Home postcode match: " + str(sum(TP_Fife65)))
print("Tayside patients with any Care Home postcode match: " + str(sum(TP_Tayside)))
print("Tayside (over 65) patients with any Care Home postcode match: " + str(sum(TP_Tayside65)))
print()

# Recall, Precision and F1 measures for the baseline
F1 = lambda p, r: 2*p*r/(p+r)
Recall_Fife = sum(TP_Fife)/sum(Fife_labels)
Precision_Fife = sum(TP_Fife)/sum(Fife_CareHome_Fife)
F1_Fife = F1(Precision_Fife, Recall_Fife)
Recall_Fife65 = sum(TP_Fife65)/sum(Fife65_labels)
Precision_Fife65 = sum(TP_Fife65)/sum(Fife65_CareHome_Fife)
F1_Fife65 = F1(Precision_Fife65, Recall_Fife65)
Recall_Tayside = sum(TP_Tayside)/sum(Tayside_labels)
Precision_Tayside = sum(TP_Tayside)/sum(Tayside_CareHome_Tayside)
F1_Tayside = F1(Precision_Tayside, Recall_Tayside)
Recall_Tayside65 = sum(TP_Tayside65)/sum(Tayside65_labels)
Precision_Tayside65 = sum(TP_Tayside65)/sum(Tayside65_CareHome_Tayside)
F1_Tayside65 = F1(Precision_Tayside65, Recall_Tayside65)
print("Fife baseline R|P|F1: " + str(sum(TP_Fife)) + "/" + str(sum(Fife_labels)) + " [" + str(round(Recall_Fife*100,2)) + "%]| " + str(sum(TP_Fife)) + "/" + str(sum(Fife_CareHome_Fife)) + " [" + str(round(Precision_Fife*100,2)) + "%]|[" + str(round(F1_Fife*100,2)) + "%]")
print("Fife (over 65) baseline R|P|F1: " + str(sum(TP_Fife65)) + "/" + str(sum(Fife65_labels)) + " [" + str(round(Recall_Fife65*100,2)) + "%]| " + str(sum(TP_Fife65)) + "/" + str(sum(Fife65_CareHome_Fife)) + " [" + str(round(Precision_Fife65*100,2)) + "%]|[" + str(round(F1_Fife65*100,2)) + "%]")
print("Tayside baseline R|P|F1: " + str(sum(TP_Tayside)) + "/" + str(sum(Tayside_labels)) + " [" + str(round(Recall_Tayside*100,2)) + "%]| " + str(sum(TP_Tayside)) + "/" + str(sum(Tayside_CareHome_Tayside)) + " [" + str(round(Precision_Tayside*100,2)) + "%]|[" + str(round(F1_Tayside*100,2)) + "%]")
print("Tayside (over 65) baseline R|P|F1: " + str(sum(TP_Tayside65)) + "/" + str(sum(Tayside65_labels)) + " [" + str(round(Recall_Tayside65*100,2)) + "%]| " + str(sum(TP_Tayside65)) + "/" + str(sum(Tayside65_CareHome_Tayside)) + " [" + str(round(Precision_Tayside65*100,2)) + "%]|[" + str(round(F1_Tayside65*100,2)) + "%]")
print()

print_results(Fife_labels, Fife_CareHome_Fife, Fife["Markov"]/100, os.path.join(folder, "Fife_Markov" + str(Markov_threshold) + "_prediction"))
print_results(Fife65_labels, Fife65_CareHome_Fife, Fife65["Markov"]/100, os.path.join(folder, "Fife65_Markov" + str(Markov_threshold) + "_prediction"))
print_results(Tayside_labels, Tayside_CareHome_Tayside, Tayside["Markov"]/100, os.path.join(folder, "Tayside_Markov" + str(Markov_threshold) + "_prediction"))
print_results(Tayside65_labels, Tayside65_CareHome_Tayside, Tayside65["Markov"]/100, os.path.join(folder, "Tayside65_Markov" + str(Markov_threshold) + "_prediction"))
print_results(Fife_labels.loc[Fife_labels.index.repeat(Fife_population)], Fife_CareHome_Fife.loc[Fife_CareHome_Fife.index.repeat(Fife_population)], Fife["Markov"].loc[Fife["Markov"].index.repeat(Fife_population)]/100, os.path.join(folder, "Fife_Markov" + str(Markov_threshold) + "_prediction_population"))
print_results(Fife65_labels.loc[Fife65_labels.index.repeat(Fife65_population)], Fife65_CareHome_Fife.loc[Fife65_CareHome_Fife.index.repeat(Fife65_population)], Fife65["Markov"].loc[Fife65["Markov"].index.repeat(Fife65_population)]/100, os.path.join(folder, "Fife65_Markov" + str(Markov_threshold) + "_prediction_population"))
print_results(Tayside_labels.loc[Tayside_labels.index.repeat(Tayside_population)], Tayside_CareHome_Tayside.loc[Tayside_CareHome_Tayside.index.repeat(Tayside_population)], Tayside["Markov"].loc[Tayside["Markov"].index.repeat(Tayside_population)]/100, os.path.join(folder, "Tayside_Markov" + str(Markov_threshold) + "_prediction_population"))
print_results(Tayside65_labels.loc[Tayside65_labels.index.repeat(Tayside65_population)], Tayside65_CareHome_Tayside.loc[Tayside65_CareHome_Tayside.index.repeat(Tayside65_population)], Tayside65["Markov"].loc[Tayside65["Markov"].index.repeat(Tayside65_population)]/100, os.path.join(folder, "Tayside65_Markov" + str(Markov_threshold) + "_prediction_population"))

print("----BASELINE (Markov score)----")
thresholds = pd.concat([Fife["Markov"],Fife_train["Markov"],Tayside["Markov"],Tayside_train["Markov"]]).sort_values().unique()#range(0,101)

labels = [Fife_train["Markov"]>=threshold for threshold in thresholds]
Fife_train_labels = Fife_train["CareHomeYN"]=="Y"
precision, recall, f1, _ = map(list, zip(*[precision_recall_fscore_support(Fife_train_labels, label, average="binary") for label in labels]))
criteria = f1
max_criteria = max(criteria)
argmax_criteria = criteria.index(max_criteria)
Markov_threshold_Fife = thresholds[argmax_criteria]
labels = labels[argmax_criteria]
plot_thresholds(precision, recall, f1, thresholds, Markov_threshold_Fife, max_criteria, os.path.join(folder, "Fife_Markov_train"))
plot_scores(Fife_train["Markov"], Fife_train_labels, labels, Markov_threshold_Fife, os.path.join(folder, "Fife_Markov_train"))
print_results(Fife_train_labels, labels, Fife_train["Markov"]/100, os.path.join(folder, "Fife_Markov_train"))

labels = [Fife65_train["Markov"]>=threshold for threshold in thresholds]
Fife65_train_labels = Fife65_train["CareHomeYN"]=="Y"
precision, recall, f1, _ = map(list, zip(*[precision_recall_fscore_support(Fife65_train_labels, label, average="binary") for label in labels]))
criteria = f1
max_criteria = max(criteria)
argmax_criteria = criteria.index(max_criteria)
Markov_threshold_Fife65 = thresholds[argmax_criteria]
labels = labels[argmax_criteria]
plot_thresholds(precision, recall, f1, thresholds, Markov_threshold_Fife65, max_criteria, os.path.join(folder, "Fife65_Markov_train"))
plot_scores(Fife65_train["Markov"], Fife65_train_labels, labels, Markov_threshold_Fife65, os.path.join(folder, "Fife65_Markov_train"))
print_results(Fife65_train_labels, labels, Fife65_train["Markov"]/100, os.path.join(folder, "Fife65_Markov_train"))

labels = [Tayside_train["Markov"]>=threshold for threshold in thresholds]
Tayside_train_labels = Tayside_train["CareHomeYN"]=="Y"
precision, recall, f1, _ = map(list, zip(*[precision_recall_fscore_support(Tayside_train_labels, label, average="binary") for label in labels]))
criteria = f1
max_criteria = max(criteria)
argmax_criteria = criteria.index(max_criteria)
Markov_threshold_Tayside = thresholds[argmax_criteria]
labels = labels[argmax_criteria]
plot_thresholds(precision, recall, f1, thresholds, Markov_threshold_Tayside, max_criteria, os.path.join(folder, "Tayside_Markov_train"))
plot_scores(Tayside_train["Markov"], Tayside_train_labels, labels, Markov_threshold_Tayside, os.path.join(folder, "Tayside_Markov_train"))
print_results(Tayside_train_labels, labels, Tayside_train["Markov"]/100, os.path.join(folder, "Tayside_Markov_train"))

labels = [Tayside65_train["Markov"]>=threshold for threshold in thresholds]
Tayside65_train_labels = Tayside65_train["CareHomeYN"]=="Y"
precision, recall, f1, _ = map(list, zip(*[precision_recall_fscore_support(Tayside65_train_labels, label, average="binary") for label in labels]))
criteria = f1
max_criteria = max(criteria)
argmax_criteria = criteria.index(max_criteria)
Markov_threshold_Tayside65 = thresholds[argmax_criteria]
labels = labels[argmax_criteria]
plot_thresholds(precision, recall, f1, thresholds, Markov_threshold_Tayside65, max_criteria, os.path.join(folder, "Tayside65_Markov_train"))
plot_scores(Tayside65_train["Markov"], Tayside65_train_labels, labels, Markov_threshold_Tayside65, os.path.join(folder, "Tayside65_Markov_train"))
print_results(Tayside65_train_labels, labels, Tayside65_train["Markov"]/100, os.path.join(folder, "Tayside65_Markov_train"))

# Patient postcodes occurrence in Care Home list
Fife_CareHome_Fife = Fife["Markov"]>=Markov_threshold_Fife
Fife65_CareHome_Fife = Fife65["Markov"]>=Markov_threshold_Fife65
Tayside_CareHome_Tayside = Tayside["Markov"]>=Markov_threshold_Tayside
Tayside65_CareHome_Tayside = Tayside65["Markov"]>=Markov_threshold_Tayside65
print("Fife Markov in Fife Care Home: " + str(sum(Fife_CareHome_Fife)) + "/" + str(len(Fife)) + " [" + str(round((sum(Fife_CareHome_Fife)/len(Fife))*100,2)) + "%]")
print("Fife (over 65) Markov in Fife Care Home: " + str(sum(Fife65_CareHome_Fife)) + "/" + str(len(Fife65)) + " [" + str(round((sum(Fife65_CareHome_Fife)/len(Fife65))*100,2)) + "%]")
print("Tayside Markov in Tayside Care Home: " + str(sum(Tayside_CareHome_Tayside)) + "/" + str(len(Tayside)) + " [" + str(round((sum(Tayside_CareHome_Tayside)/len(Tayside))*100,2)) + "%]")
print("Tayside (over 65) Markov in Tayside Care Home: " + str(sum(Tayside65_CareHome_Tayside)) + "/" + str(len(Tayside65)) + " [" + str(round((sum(Tayside65_CareHome_Tayside)/len(Tayside65))*100,2)) + "%]")
print()

# Baseline considering any postcode match as Care Home patient
TP_Fife = Fife_labels&Fife_CareHome_Fife
TP_Fife65 = Fife65_labels&Fife65_CareHome_Fife
TP_Tayside = Tayside_labels&Tayside_CareHome_Tayside
TP_Tayside65 = Tayside65_labels&Tayside65_CareHome_Tayside
print("Fife patients with any Care Home postcode match: " + str(sum(TP_Fife)))
print("Fife (over 65) patients with any Care Home postcode match: " + str(sum(TP_Fife65)))
print("Tayside patients with any Care Home postcode match: " + str(sum(TP_Tayside)))
print("Tayside (over 65) patients with any Care Home postcode match: " + str(sum(TP_Tayside65)))
print()

# Recall, Precision and F1 measures for the baseline
F1 = lambda p, r: 2*p*r/(p+r)
Recall_Fife = sum(TP_Fife)/sum(Fife_labels)
Precision_Fife = sum(TP_Fife)/sum(Fife_CareHome_Fife)
F1_Fife = F1(Precision_Fife, Recall_Fife)
Recall_Fife65 = sum(TP_Fife65)/sum(Fife65_labels)
Precision_Fife65 = sum(TP_Fife65)/sum(Fife65_CareHome_Fife)
F1_Fife65 = F1(Precision_Fife65, Recall_Fife65)
Recall_Tayside = sum(TP_Tayside)/sum(Tayside_labels)
Precision_Tayside = sum(TP_Tayside)/sum(Tayside_CareHome_Tayside)
F1_Tayside = F1(Precision_Tayside, Recall_Tayside)
Recall_Tayside65 = sum(TP_Tayside65)/sum(Tayside65_labels)
Precision_Tayside65 = sum(TP_Tayside65)/sum(Tayside65_CareHome_Tayside)
F1_Tayside65 = F1(Precision_Tayside65, Recall_Tayside65)
print("Fife baseline R|P|F1: " + str(sum(TP_Fife)) + "/" + str(sum(Fife_labels)) + " [" + str(round(Recall_Fife*100,2)) + "%]| " + str(sum(TP_Fife)) + "/" + str(sum(Fife_CareHome_Fife)) + " [" + str(round(Precision_Fife*100,2)) + "%]|[" + str(round(F1_Fife*100,2)) + "%]")
print("Fife (over 65) baseline R|P|F1: " + str(sum(TP_Fife65)) + "/" + str(sum(Fife65_labels)) + " [" + str(round(Recall_Fife65*100,2)) + "%]| " + str(sum(TP_Fife65)) + "/" + str(sum(Fife65_CareHome_Fife)) + " [" + str(round(Precision_Fife65*100,2)) + "%]|[" + str(round(F1_Fife65*100,2)) + "%]")
print("Tayside baseline R|P|F1: " + str(sum(TP_Tayside)) + "/" + str(sum(Tayside_labels)) + " [" + str(round(Recall_Tayside*100,2)) + "%]| " + str(sum(TP_Tayside)) + "/" + str(sum(Tayside_CareHome_Tayside)) + " [" + str(round(Precision_Tayside*100,2)) + "%]|[" + str(round(F1_Tayside*100,2)) + "%]")
print("Tayside (over 65) baseline R|P|F1: " + str(sum(TP_Tayside65)) + "/" + str(sum(Tayside65_labels)) + " [" + str(round(Recall_Tayside65*100,2)) + "%]| " + str(sum(TP_Tayside65)) + "/" + str(sum(Tayside65_CareHome_Tayside)) + " [" + str(round(Precision_Tayside65*100,2)) + "%]|[" + str(round(F1_Tayside65*100,2)) + "%]")
print()

print_results(Fife_labels, Fife_CareHome_Fife, Fife["Markov"]/100, os.path.join(folder, "Fife_Markov_prediction"))
print_results(Fife65_labels, Fife65_CareHome_Fife, Fife65["Markov"]/100, os.path.join(folder, "Fife65_Markov_prediction"))
print_results(Tayside_labels, Tayside_CareHome_Tayside, Tayside["Markov"]/100, os.path.join(folder, "Tayside_Markov_prediction"))
print_results(Tayside65_labels, Tayside65_CareHome_Tayside, Tayside65["Markov"]/100, os.path.join(folder, "Tayside65_Markov_prediction"))
print_results(Fife_labels.loc[Fife_labels.index.repeat(Fife_population)], Fife_CareHome_Fife.loc[Fife_CareHome_Fife.index.repeat(Fife_population)], Fife["Markov"].loc[Fife["Markov"].index.repeat(Fife_population)]/100, os.path.join(folder, "Fife_Markov_prediction_population"))
print_results(Fife65_labels.loc[Fife65_labels.index.repeat(Fife65_population)], Fife65_CareHome_Fife.loc[Fife65_CareHome_Fife.index.repeat(Fife65_population)], Fife65["Markov"].loc[Fife65["Markov"].index.repeat(Fife65_population)]/100, os.path.join(folder, "Fife65_Markov_prediction_population"))
print_results(Tayside_labels.loc[Tayside_labels.index.repeat(Tayside_population)], Tayside_CareHome_Tayside.loc[Tayside_CareHome_Tayside.index.repeat(Tayside_population)], Tayside["Markov"].loc[Tayside["Markov"].index.repeat(Tayside_population)]/100, os.path.join(folder, "Tayside_Markov_prediction_population"))
print_results(Tayside65_labels.loc[Tayside65_labels.index.repeat(Tayside65_population)], Tayside65_CareHome_Tayside.loc[Tayside65_CareHome_Tayside.index.repeat(Tayside65_population)], Tayside65["Markov"].loc[Tayside65["Markov"].index.repeat(Tayside65_population)]/100, os.path.join(folder, "Tayside65_Markov_prediction_population"))

Phonix_threshold = 13
print("----BASELINE (Phonix score [cutoff=" + str(Phonix_threshold) + "])----")
# Patient postcodes occurrence in Care Home list
Fife_CareHome_Fife = Fife["Phonix"]>=Phonix_threshold
Fife65_CareHome_Fife = Fife65["Phonix"]>=Phonix_threshold
Tayside_CareHome_Tayside = Tayside["Phonix"]>=Phonix_threshold
Tayside65_CareHome_Tayside = Tayside65["Phonix"]>=Phonix_threshold
print("Fife Phonix in Fife Care Home: " + str(sum(Fife_CareHome_Fife)) + "/" + str(len(Fife)) + " [" + str(round((sum(Fife_CareHome_Fife)/len(Fife))*100,2)) + "%]")
print("Fife (over 65) Phonix in Fife Care Home: " + str(sum(Fife65_CareHome_Fife)) + "/" + str(len(Fife65)) + " [" + str(round((sum(Fife65_CareHome_Fife)/len(Fife65))*100,2)) + "%]")
print("Tayside Phonix in Tayside Care Home: " + str(sum(Tayside_CareHome_Tayside)) + "/" + str(len(Tayside)) + " [" + str(round((sum(Tayside_CareHome_Tayside)/len(Tayside))*100,2)) + "%]")
print("Tayside (over 65) Phonix in Tayside Care Home: " + str(sum(Tayside65_CareHome_Tayside)) + "/" + str(len(Tayside65)) + " [" + str(round((sum(Tayside65_CareHome_Tayside)/len(Tayside65))*100,2)) + "%]")
print()

# Baseline considering any postcode match as Care Home patient
TP_Fife = Fife_labels&Fife_CareHome_Fife
TP_Fife65 = Fife65_labels&Fife65_CareHome_Fife
TP_Tayside = Tayside_labels&Tayside_CareHome_Tayside
TP_Tayside65 = Tayside65_labels&Tayside65_CareHome_Tayside
print("Fife patients with any Care Home postcode match: " + str(sum(TP_Fife)))
print("Fife (over 65) patients with any Care Home postcode match: " + str(sum(TP_Fife65)))
print("Tayside patients with any Care Home postcode match: " + str(sum(TP_Tayside)))
print("Tayside (over 65) patients with any Care Home postcode match: " + str(sum(TP_Tayside65)))
print()

# Recall, Precision and F1 measures for the baseline
F1 = lambda p, r: 2*p*r/(p+r)
Recall_Fife = sum(TP_Fife)/sum(Fife_labels)
Precision_Fife = sum(TP_Fife)/sum(Fife_CareHome_Fife)
F1_Fife = F1(Precision_Fife, Recall_Fife)
Recall_Fife65 = sum(TP_Fife65)/sum(Fife65_labels)
Precision_Fife65 = sum(TP_Fife65)/sum(Fife65_CareHome_Fife)
F1_Fife65 = F1(Precision_Fife65, Recall_Fife65)
Recall_Tayside = sum(TP_Tayside)/sum(Tayside_labels)
Precision_Tayside = sum(TP_Tayside)/sum(Tayside_CareHome_Tayside)
F1_Tayside = F1(Precision_Tayside, Recall_Tayside)
Recall_Tayside65 = sum(TP_Tayside65)/sum(Tayside65_labels)
Precision_Tayside65 = sum(TP_Tayside65)/sum(Tayside65_CareHome_Tayside)
F1_Tayside65 = F1(Precision_Tayside65, Recall_Tayside65)
print("Fife baseline R|P|F1: " + str(sum(TP_Fife)) + "/" + str(sum(Fife_labels)) + " [" + str(round(Recall_Fife*100,2)) + "%]| " + str(sum(TP_Fife)) + "/" + str(sum(Fife_CareHome_Fife)) + " [" + str(round(Precision_Fife*100,2)) + "%]|[" + str(round(F1_Fife*100,2)) + "%]")
print("Fife (over 65) baseline R|P|F1: " + str(sum(TP_Fife65)) + "/" + str(sum(Fife65_labels)) + " [" + str(round(Recall_Fife65*100,2)) + "%]| " + str(sum(TP_Fife65)) + "/" + str(sum(Fife65_CareHome_Fife)) + " [" + str(round(Precision_Fife65*100,2)) + "%]|[" + str(round(F1_Fife65*100,2)) + "%]")
print("Tayside baseline R|P|F1: " + str(sum(TP_Tayside)) + "/" + str(sum(Tayside_labels)) + " [" + str(round(Recall_Tayside*100,2)) + "%]| " + str(sum(TP_Tayside)) + "/" + str(sum(Tayside_CareHome_Tayside)) + " [" + str(round(Precision_Tayside*100,2)) + "%]|[" + str(round(F1_Tayside*100,2)) + "%]")
print("Tayside (over 65) baseline R|P|F1: " + str(sum(TP_Tayside65)) + "/" + str(sum(Tayside65_labels)) + " [" + str(round(Recall_Tayside65*100,2)) + "%]| " + str(sum(TP_Tayside65)) + "/" + str(sum(Tayside65_CareHome_Tayside)) + " [" + str(round(Precision_Tayside65*100,2)) + "%]|[" + str(round(F1_Tayside65*100,2)) + "%]")
print()

print_results(Fife_labels, Fife_CareHome_Fife, Fife["Phonix"]/100, os.path.join(folder, "Fife_Phonix" + str(Phonix_threshold) + "_prediction"))
print_results(Fife65_labels, Fife65_CareHome_Fife, Fife65["Phonix"]/100, os.path.join(folder, "Fife65_Phonix" + str(Phonix_threshold) + "_prediction"))
print_results(Tayside_labels, Tayside_CareHome_Tayside, Tayside["Phonix"]/100, os.path.join(folder, "Tayside_Phonix" + str(Phonix_threshold) + "_prediction"))
print_results(Tayside65_labels, Tayside65_CareHome_Tayside, Tayside65["Phonix"]/100, os.path.join(folder, "Tayside65_Phonix" + str(Phonix_threshold) + "_prediction"))
print_results(Fife_labels.loc[Fife_labels.index.repeat(Fife_population)], Fife_CareHome_Fife.loc[Fife_CareHome_Fife.index.repeat(Fife_population)], Fife["Phonix"].loc[Fife["Phonix"].index.repeat(Fife_population)]/100, os.path.join(folder, "Fife_Phonix" + str(Phonix_threshold) + "_prediction_population"))
print_results(Fife65_labels.loc[Fife65_labels.index.repeat(Fife65_population)], Fife65_CareHome_Fife.loc[Fife65_CareHome_Fife.index.repeat(Fife65_population)], Fife65["Phonix"].loc[Fife65["Phonix"].index.repeat(Fife65_population)]/100, os.path.join(folder, "Fife65_Phonix" + str(Phonix_threshold) + "_prediction_population"))
print_results(Tayside_labels.loc[Tayside_labels.index.repeat(Tayside_population)], Tayside_CareHome_Tayside.loc[Tayside_CareHome_Tayside.index.repeat(Tayside_population)], Tayside["Phonix"].loc[Tayside["Phonix"].index.repeat(Tayside_population)]/100, os.path.join(folder, "Tayside_Phonix" + str(Phonix_threshold) + "_prediction_population"))
print_results(Tayside65_labels.loc[Tayside65_labels.index.repeat(Tayside65_population)], Tayside65_CareHome_Tayside.loc[Tayside65_CareHome_Tayside.index.repeat(Tayside65_population)], Tayside65["Phonix"].loc[Tayside65["Phonix"].index.repeat(Tayside65_population)]/100, os.path.join(folder, "Tayside65_Phonix" + str(Phonix_threshold) + "_prediction_population"))

print("----BASELINE (Phonix score)----")
thresholds = pd.concat([Fife["Phonix"],Fife_train["Phonix"],Tayside["Phonix"],Tayside_train["Phonix"]]).sort_values().unique()#range(0,101)

labels = [Fife_train["Phonix"]>=threshold for threshold in thresholds]
Fife_train_labels = Fife_train["CareHomeYN"]=="Y"
precision, recall, f1, _ = map(list, zip(*[precision_recall_fscore_support(Fife_train_labels, label, average="binary") for label in labels]))
criteria = f1
max_criteria = max(criteria)
argmax_criteria = criteria.index(max_criteria)
Phonix_threshold_Fife = thresholds[argmax_criteria]
labels = labels[argmax_criteria]
plot_thresholds(precision, recall, f1, thresholds, Phonix_threshold_Fife, max_criteria, os.path.join(folder, "Fife_Phonix_train"))
plot_scores(Fife_train["Phonix"], Fife_train_labels, labels, Phonix_threshold_Fife, os.path.join(folder, "Fife_Phonix_train"))
print_results(Fife_train_labels, labels, Fife_train["Phonix"]/100, os.path.join(folder, "Fife_Phonix_train"))

labels = [Fife65_train["Phonix"]>=threshold for threshold in thresholds]
Fife65_train_labels = Fife65_train["CareHomeYN"]=="Y"
precision, recall, f1, _ = map(list, zip(*[precision_recall_fscore_support(Fife65_train_labels, label, average="binary") for label in labels]))
criteria = f1
max_criteria = max(criteria)
argmax_criteria = criteria.index(max_criteria)
Phonix_threshold_Fife65 = thresholds[argmax_criteria]
labels = labels[argmax_criteria]
plot_thresholds(precision, recall, f1, thresholds, Phonix_threshold_Fife65, max_criteria, os.path.join(folder, "Fife65_Phonix_train"))
plot_scores(Fife65_train["Phonix"], Fife65_train_labels, labels, Phonix_threshold_Fife65, os.path.join(folder, "Fife65_Phonix_train"))
print_results(Fife65_train_labels, labels, Fife65_train["Phonix"]/100, os.path.join(folder, "Fife65_Phonix_train"))

labels = [Tayside_train["Phonix"]>=threshold for threshold in thresholds]
Tayside_train_labels = Tayside_train["CareHomeYN"]=="Y"
precision, recall, f1, _ = map(list, zip(*[precision_recall_fscore_support(Tayside_train_labels, label, average="binary") for label in labels]))
criteria = f1
max_criteria = max(criteria)
argmax_criteria = criteria.index(max_criteria)
Phonix_threshold_Tayside = thresholds[argmax_criteria]
labels = labels[argmax_criteria]
plot_thresholds(precision, recall, f1, thresholds, Phonix_threshold_Tayside, max_criteria, os.path.join(folder, "Tayside_Phonix_train"))
plot_scores(Tayside_train["Phonix"], Tayside_train_labels, labels, Phonix_threshold_Tayside, os.path.join(folder, "Tayside_Phonix_train"))
print_results(Tayside_train_labels, labels, Tayside_train["Phonix"]/100, os.path.join(folder, "Tayside_Phonix_train"))

labels = [Tayside65_train["Phonix"]>=threshold for threshold in thresholds]
Tayside65_train_labels = Tayside65_train["CareHomeYN"]=="Y"
precision, recall, f1, _ = map(list, zip(*[precision_recall_fscore_support(Tayside65_train_labels, label, average="binary") for label in labels]))
criteria = f1
max_criteria = max(criteria)
argmax_criteria = criteria.index(max_criteria)
Phonix_threshold_Tayside65 = thresholds[argmax_criteria]
labels = labels[argmax_criteria]
plot_thresholds(precision, recall, f1, thresholds, Phonix_threshold_Tayside65, max_criteria, os.path.join(folder, "Tayside65_Phonix_train"))
plot_scores(Tayside65_train["Phonix"], Tayside65_train_labels, labels, Phonix_threshold_Tayside65, os.path.join(folder, "Tayside65_Phonix_train"))
print_results(Tayside65_train_labels, labels, Tayside65_train["Phonix"]/100, os.path.join(folder, "Tayside65_Phonix_train"))

# Patient postcodes occurrence in Care Home list
Fife_CareHome_Fife = Fife["Phonix"]>=Phonix_threshold_Fife
Fife65_CareHome_Fife = Fife65["Phonix"]>=Phonix_threshold_Fife65
Tayside_CareHome_Tayside = Tayside["Phonix"]>=Phonix_threshold_Tayside
Tayside65_CareHome_Tayside = Tayside65["Phonix"]>=Phonix_threshold_Tayside65
print("Fife Phonix in Fife Care Home: " + str(sum(Fife_CareHome_Fife)) + "/" + str(len(Fife)) + " [" + str(round((sum(Fife_CareHome_Fife)/len(Fife))*100,2)) + "%]")
print("Fife (over 65) Phonix in Fife Care Home: " + str(sum(Fife65_CareHome_Fife)) + "/" + str(len(Fife65)) + " [" + str(round((sum(Fife65_CareHome_Fife)/len(Fife65))*100,2)) + "%]")
print("Tayside Phonix in Tayside Care Home: " + str(sum(Tayside_CareHome_Tayside)) + "/" + str(len(Tayside)) + " [" + str(round((sum(Tayside_CareHome_Tayside)/len(Tayside))*100,2)) + "%]")
print("Tayside (over 65) Phonix in Tayside Care Home: " + str(sum(Tayside65_CareHome_Tayside)) + "/" + str(len(Tayside65)) + " [" + str(round((sum(Tayside65_CareHome_Tayside)/len(Tayside65))*100,2)) + "%]")
print()

# Baseline considering any postcode match as Care Home patient
TP_Fife = Fife_labels&Fife_CareHome_Fife
TP_Fife65 = Fife65_labels&Fife65_CareHome_Fife
TP_Tayside = Tayside_labels&Tayside_CareHome_Tayside
TP_Tayside65 = Tayside65_labels&Tayside65_CareHome_Tayside
print("Fife patients with any Care Home postcode match: " + str(sum(TP_Fife)))
print("Fife (over 65) patients with any Care Home postcode match: " + str(sum(TP_Fife65)))
print("Tayside patients with any Care Home postcode match: " + str(sum(TP_Tayside)))
print("Tayside (over 65) patients with any Care Home postcode match: " + str(sum(TP_Tayside65)))
print()

# Recall, Precision and F1 measures for the baseline
F1 = lambda p, r: 2*p*r/(p+r)
Recall_Fife = sum(TP_Fife)/sum(Fife_labels)
Precision_Fife = sum(TP_Fife)/sum(Fife_CareHome_Fife)
F1_Fife = F1(Precision_Fife, Recall_Fife)
Recall_Fife65 = sum(TP_Fife65)/sum(Fife65_labels)
Precision_Fife65 = sum(TP_Fife65)/sum(Fife65_CareHome_Fife)
F1_Fife65 = F1(Precision_Fife65, Recall_Fife65)
Recall_Tayside = sum(TP_Tayside)/sum(Tayside_labels)
Precision_Tayside = sum(TP_Tayside)/sum(Tayside_CareHome_Tayside)
F1_Tayside = F1(Precision_Tayside, Recall_Tayside)
Recall_Tayside65 = sum(TP_Tayside65)/sum(Tayside65_labels)
Precision_Tayside65 = sum(TP_Tayside65)/sum(Tayside65_CareHome_Tayside)
F1_Tayside65 = F1(Precision_Tayside65, Recall_Tayside65)
print("Fife baseline R|P|F1: " + str(sum(TP_Fife)) + "/" + str(sum(Fife_labels)) + " [" + str(round(Recall_Fife*100,2)) + "%]| " + str(sum(TP_Fife)) + "/" + str(sum(Fife_CareHome_Fife)) + " [" + str(round(Precision_Fife*100,2)) + "%]|[" + str(round(F1_Fife*100,2)) + "%]")
print("Fife (over 65) baseline R|P|F1: " + str(sum(TP_Fife65)) + "/" + str(sum(Fife65_labels)) + " [" + str(round(Recall_Fife65*100,2)) + "%]| " + str(sum(TP_Fife65)) + "/" + str(sum(Fife65_CareHome_Fife)) + " [" + str(round(Precision_Fife65*100,2)) + "%]|[" + str(round(F1_Fife65*100,2)) + "%]")
print("Tayside baseline R|P|F1: " + str(sum(TP_Tayside)) + "/" + str(sum(Tayside_labels)) + " [" + str(round(Recall_Tayside*100,2)) + "%]| " + str(sum(TP_Tayside)) + "/" + str(sum(Tayside_CareHome_Tayside)) + " [" + str(round(Precision_Tayside*100,2)) + "%]|[" + str(round(F1_Tayside*100,2)) + "%]")
print("Tayside (over 65) baseline R|P|F1: " + str(sum(TP_Tayside65)) + "/" + str(sum(Tayside65_labels)) + " [" + str(round(Recall_Tayside65*100,2)) + "%]| " + str(sum(TP_Tayside65)) + "/" + str(sum(Tayside65_CareHome_Tayside)) + " [" + str(round(Precision_Tayside65*100,2)) + "%]|[" + str(round(F1_Tayside65*100,2)) + "%]")
print()

print_results(Fife_labels, Fife_CareHome_Fife, Fife["Phonix"]/100, os.path.join(folder, "Fife_Phonix_prediction"))
print_results(Fife65_labels, Fife65_CareHome_Fife, Fife65["Phonix"]/100, os.path.join(folder, "Fife65_Phonix_prediction"))
print_results(Tayside_labels, Tayside_CareHome_Tayside, Tayside["Phonix"]/100, os.path.join(folder, "Tayside_Phonix_prediction"))
print_results(Tayside65_labels, Tayside65_CareHome_Tayside, Tayside65["Phonix"]/100, os.path.join(folder, "Tayside65_Phonix_prediction"))
print_results(Fife_labels.loc[Fife_labels.index.repeat(Fife_population)], Fife_CareHome_Fife.loc[Fife_CareHome_Fife.index.repeat(Fife_population)], Fife["Phonix"].loc[Fife["Phonix"].index.repeat(Fife_population)]/100, os.path.join(folder, "Fife_Phonix_prediction_population"))
print_results(Fife65_labels.loc[Fife65_labels.index.repeat(Fife65_population)], Fife65_CareHome_Fife.loc[Fife65_CareHome_Fife.index.repeat(Fife65_population)], Fife65["Phonix"].loc[Fife65["Phonix"].index.repeat(Fife65_population)]/100, os.path.join(folder, "Fife65_Phonix_prediction_population"))
print_results(Tayside_labels.loc[Tayside_labels.index.repeat(Tayside_population)], Tayside_CareHome_Tayside.loc[Tayside_CareHome_Tayside.index.repeat(Tayside_population)], Tayside["Phonix"].loc[Tayside["Phonix"].index.repeat(Tayside_population)]/100, os.path.join(folder, "Tayside_Phonix_prediction_population"))
print_results(Tayside65_labels.loc[Tayside65_labels.index.repeat(Tayside65_population)], Tayside65_CareHome_Tayside.loc[Tayside65_CareHome_Tayside.index.repeat(Tayside65_population)], Tayside65["Phonix"].loc[Tayside65["Phonix"].index.repeat(Tayside65_population)]/100, os.path.join(folder, "Tayside65_Phonix_prediction_population"))
