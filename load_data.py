import os
import pandas as pd

# Load data of Patient addresses
patients = pd.read_csv("P:/Project 3410 - ESRC AMR - Carehome Antibiotic Use/AddressCounts/number_of_people_at_each_address_all_addresses_at_20200430.txt", delimiter="|", keep_default_na=False)

patients2021 = pd.read_excel("P:/Project 3410 - ESRC AMR - Carehome Antibiotic Use/2020_Addresses/Victor/addresses_annotation_2021.xlsx", sheet_name="missing addresses bg", keep_default_na=False)
patients = pd.merge(patients, patients2021[["ARCHaddrID","BG","CSNumber"]], on="ARCHaddrID", how="left")
patients["CareHomeYN"] = patients["manually_determined_care_home_flag"]
patients.loc[patients["manually_determined_care_home_flag"]=="C","CareHomeYN"] = "N"
patients.loc[(patients["manually_determined_care_home_flag"]=="M")&(patients["Markov"]==0),"CareHomeYN"] = "N"
patients.loc[patients["BG"].isin(["U","OLD"]),"CareHomeYN"] = "N"
patients.loc[(patients["manually_determined_care_home_flag"]=="M")&(patients["Markov"]!=0),"CareHomeYN"] = patients["BG"]
patients.loc[(patients["manually_determined_care_home_flag"]=="M")&(patients["Markov"]!=0),"ConsensusCSNumber"] = patients["CSNumber"]
patients = patients[["ARCHaddrID", "current_address_L1", "current_address_L2", "current_address_L3", "current_address_L4", "current_postcode", "Markov", "Phonix", "patients_under_65", "patients_65_or_over", "CareHomeYN","ConsensusCSNumber"]]

# Load data of Care Home addresses
carehomes = pd.read_excel("P:/Project 3410 - ESRC AMR - Carehome Antibiotic Use/2020_Addresses/CareInspectorate2020/Care_Homes_30_April_2020.xlsx", sheet_name="Care_Homes_30_April_2020", keep_default_na=False)
carehomes = carehomes[["CSNumber", "ServiceName", "Address_line_1", "Address_line_2", "Address_line_3", "Service_town", "Service_Postcode", "ServiceStatus", "Subtype", "Health_Board_Name"]]

# Save data of Patient and Care Home addresses
if not os.path.exists("Addresses_2021.xlsx"):
    with pd.ExcelWriter("Addresses_2021.xlsx") as writer:
        patients.to_excel(writer, sheet_name="Patient addresses", index=False)
        carehomes.to_excel(writer, sheet_name="Care Home addresses", index=False)

# Save data of Patient and Care Home addresses in Fife
if not os.path.exists("Addresses_Fife_2021.xlsx"):
    Fife = patients[patients["ARCHaddrID"].str[-1]=="F"]
    Fife65 = Fife[Fife["patients_65_or_over"]!=0]
    CareHome_Fife = carehomes[carehomes["Health_Board_Name"]=="Fife"].drop(["Health_Board_Name"], axis=1)
    with pd.ExcelWriter("Addresses_Fife_2021.xlsx") as writer:
        Fife.to_excel(writer, sheet_name="Patient addresses", index=False)
        Fife65.to_excel(writer, sheet_name="Patient over 65 addresses", index=False)
        CareHome_Fife.to_excel(writer, sheet_name="Care Home addresses", index=False)

# Save data of Patient and Care Home addresses in Tayside
if not os.path.exists("Addresses_Tayside_2021.xlsx"):
    Tayside = patients[patients["ARCHaddrID"].str[-1]=="T"]
    Tayside65 = Tayside[Tayside["patients_65_or_over"]!=0]
    CareHome_Tayside = carehomes[carehomes["Health_Board_Name"]=="Tayside"].drop(["Health_Board_Name"], axis=1)
    with pd.ExcelWriter("Addresses_Tayside_2021.xlsx") as writer:
        Tayside.to_excel(writer, sheet_name="Patient addresses", index=False)
        Tayside65.to_excel(writer, sheet_name="Patient over 65 addresses", index=False)
        CareHome_Tayside.to_excel(writer, sheet_name="Care Home addresses", index=False)

"""
# patients with cancelled CSNumber
pset = set(patients["ConsensusCSNumber"].unique())
chset = set(carehomes["CSNumber"].unique())
diff = pset.difference(chset)
diff.remove('')
missing = patients[(patients["ConsensusCSNumber"].isin(diff))&(patients["CareHomeYN"]=="Y")]
cancelled = missing["ConsensusCSNumber"].unique()


noPC = patients[(patients["current_postcode"]=="")&(patients["CareHomeYN"]!="N")]

# patients with CSNumber, not being "Y"
patients[(patients["ConsensusCSNumber"]!="")&(patients["CareHomeYN"]!="Y")]
# patients without CSNumber, being "Y"
patients[(patients["ConsensusCSNumber"]=="")&(patients["CareHomeYN"]=="Y")]


# patients with not active CSNumber
chset = set(carehomes[carehomes["ServiceStatus"]!="Active"]["CSNumber"].unique())
missing_notActive = patients[patients["ConsensusCSNumber"].isin(chset)]
notActive = carehomes[carehomes["CSNumber"].isin(missing_notActive["ConsensusCSNumber"].unique())]


# patients with not in Fife or Tayside CSNumber
chset = set(carehomes[~carehomes["Health_Board_Name"].isin(["Fife","Tayside"])]["CSNumber"].unique())
missing_notFifenotTayside = patients[patients["ConsensusCSNumber"].isin(chset)]
notFifenotTayside = carehomes[carehomes["CSNumber"].isin(missing_notFifenotTayside["ConsensusCSNumber"].unique())]


# patients with not Older People or Physical and Sensory Impairment CSNumber
chset = set(carehomes[~carehomes["Subtype"].isin(["Older People","Physical and Sensory Impairment"])]["CSNumber"].unique())
missing_notOldernotImpairment = patients[patients["ConsensusCSNumber"].isin(chset)]
notOldernotImpairment = carehomes[carehomes["CSNumber"].isin(missing_notOldernotImpairment["ConsensusCSNumber"].unique())]


# patients with carehome in the name
keywords = ["carehome", "nursinghome", "residentialhome", "reshome", "care home", "nursing home","residential home", "res home", "nursing carehome", "residential carehome", "res carehome", "nursing care home", "residential care home", "res care home", "care hm","nursing hm","residential hm", "res hm", "nursing care hm", "residential care hm", "res care hm"]
regex = "|".join([r"\b"+keyword.lower()+r"\b" for keyword in keywords])
carehome_names = patients[(patients["ConsensusCSNumber"]=="")&(patients["CareHomeYN"]!="Y")&(patients["current_address_L1"].str.lower().str.contains(regex))]
carehome_names = patients[(patients["ConsensusCSNumber"]=="")&(patients["CareHomeYN"]!="Y")&(patients["current_address_L1"].str.lower().str.contains(regex))]
"""