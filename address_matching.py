import os
import re
import itertools
from collections import Counter
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.metrics import precision_recall_fscore_support
from utils import print_results, plot_thresholds, plot_scores
from not_in_library import nan_to_num

LETTERS = [str(i) for i in range(0,9+1)] + [" "] + [chr(i) for i in range(ord('a'), ord('z')+1)]

def clean_str(string, postcode=False):
    """ Keep characters and numbers and replace multiple spaces."""
    string = re.sub(r"[^0-9a-zA-Z]", " ", string)
    if postcode:
        string = re.sub(r"\s", "", string)
    else:
        string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def text2vec(text, n_grams_range=(1,1), norm=True):
    """ Convert text into list vector of character ngrams."""
    # The character n-grams dictionary
    dictionary = {"".join(p): 0.0 for r in range(n_grams_range[0],n_grams_range[1]+1) for p in itertools.product(LETTERS, repeat=r)}
    # Convert text into character n-grams vector
    text2ngrams = dict(Counter([text[i:i+r] if len(text)>=r else text + " "*(r-len(text)) for r in range(n_grams_range[0],n_grams_range[1]+1) for i in range(len(text)-r+1 if len(text)>=r else 1)]))
    #text2ngrams = dict(Counter([text[i:i+r] for r in range(n_grams_range[0],n_grams_range[1]+1) for i in range(len(text)-r+1)]))
    dictionary.update(text2ngrams)
    vector = list(dictionary.values())
    # Vector normalization
    if norm and text:
        vector = [i/sum(vector) for i in vector]
    return vector

def address_matching_score(distance, Region, CareHome, vector=False, n_grams_range=(1,1), norm=True, inverse=False, name=""):
    """ Gives a matrix score according to a given distance between a Region list of patients and a Care Home list of addresses."""
    # Preprocessing
    drop_PC = True
    split_numbers = True
    split_only_numbers = False
    split_remove = True
    split_comma = True
    # Postprocessing
    filter_PC = True
    service_town = False
    carehome_service = True
    carehome_name = True
    
    if not os.path.exists(name + "_scores.csv"):
        # Preprocessing
        Region_data = Region.drop(["current_postcode"], axis=1) if drop_PC else Region
        Region_data = Region_data.drop(["ConsensusCSNumber"], axis=1)
        CareHome_data = CareHome.drop(["Service_Postcode"], axis=1) if drop_PC else CareHome
        CareHome_data = CareHome_data.drop(["Subtype"], axis=1)
        CareHome_data = CareHome_data.drop(["CSNumber"], axis=1)
        Region_data = Region_data.values.tolist()
        CareHome_data = CareHome_data.values.tolist()
        
        # Divide by commas
        if split_comma:
            Region_data = [[p for pline in patient for p in pline.split(",") if pline] for patient in Region_data]
            CareHome_data = [[ch for ch_line in carehome for ch in ch_line.split(",") if ch_line] for carehome in CareHome_data]

        if split_numbers:
            # Extract numbers or words with numbers separately
            regex = r"[0-9]+" if split_only_numbers else r"\S*[0-9]\S*"
            #Region_data = [[[re.sub(regex, " ", p_line) if split_remove else p_line]+list(set(re.findall(regex, p_line))) for p_line in patient if p_line] for patient in Region_data]
            #CareHome_data = [[[re.sub(regex, " ", ch_line) if split_remove else ch_line]+list(set(re.findall(regex, ch_line))) for ch_line in carehome if ch_line] for carehome in CareHome_data]
            # Separate new columns
            #Region_data = [[addr for p_line in patient for addr in p_line] for patient in Region_data]
            #CareHome_data = [[addr for ch_line in carehome for addr in ch_line] for carehome in CareHome_data]
            
            Region_numbers = [[re_line for p_line in patient for re_line in re.findall(regex, p_line) if p_line and re_line] for patient in Region_data]
            CareHome_numbers = [[re_line for ch_line in carehome for re_line in re.findall(regex, ch_line) if ch_line and re_line] for carehome in CareHome_data]
            #Region_numbers = [[re.findall(regex, p_line) for p_line in patient if p_line] for patient in Region_data]
            #CareHome_numbers = [[re.findall(regex, ch_line) for ch_line in carehome if ch_line] for carehome in CareHome_data]
            #Region_numbers = [[line for p_line in patient for line in p_line] for patient in Region_numbers]
            #CareHome_numbers = [[line for p_line in patient for line in p_line] for patient in CareHome_numbers]
            Region_numbers = [[clean_str(p_line) for p_line in patient if p_line] for patient in Region_numbers]
            CareHome_numbers = [[clean_str(ch_line) for ch_line in carehome if ch_line] for carehome in CareHome_numbers]
            
            if split_remove:
                # Remove numbers or words with numbers separately
                Region_data = [[re.sub(regex, " ", p_line) for p_line in patient if p_line] for patient in Region_data]
                CareHome_data = [[re.sub(regex, " ", ch_line) for ch_line in carehome if ch_line] for carehome in CareHome_data]
        
        # Clean text
        #Region_data = Region_data.applymap(lambda x: clean_str(x))
        #CareHome_data = CareHome_data.applymap(lambda x: clean_str(x))
        Region_data = [[clean_str(p_line) for p_line in patient if p_line and clean_str(p_line)] for patient in Region_data]
        CareHome_data = [[clean_str(ch_line) for ch_line in carehome if ch_line and clean_str(ch_line)] for carehome in CareHome_data]
        
        # Transform strings into character frequency vectors
        if vector:
            #Region_data = Region_data.applymap(lambda x: text2vec(x, n_grams_range=n_grams_range, norm=norm))
            #CareHome_data = CareHome_data.applymap(lambda x: text2vec(x, n_grams_range=n_grams_range, norm=norm))
            Region_data = [[text2vec(p_line, n_grams_range=n_grams_range, norm=norm) for p_line in patient] for patient in Region_data]
            CareHome_data = [[text2vec(ch_line, n_grams_range=n_grams_range, norm=norm) for ch_line in carehome] for carehome in CareHome_data]
            if inverse:
                # Multiply character frequency vectors by inverse document frequency
                #scores = [p_line for patient in Region_data.values.tolist() for p_line in patient]+[ch_line for carehome in CareHome_data.values.tolist() for ch_line in carehome]
                scores = [p_line for patient in Region_data for p_line in patient]+[ch_line for carehome in CareHome_data for ch_line in carehome]
                N = len(scores)
                df = np.array(scores, dtype=bool).sum(axis=0)
                idf = [0 if i==np.inf else i for i in np.log10(N/df).tolist()]
                #Region_data = Region_data.applymap(lambda x: [x[i]*idf[i] for i in range(len(x))])
                #CareHome_data = CareHome_data.applymap(lambda x: [x[i]*idf[i] for i in range(len(x))])
                Region_data = np.multiply(Region_data, idf).tolist()
                CareHome_data = np.multiply(CareHome_data, idf).tolist()
        # Transform strings into character n-grams
        elif n_grams_range[0]>1:
            r = n_grams_range[0]
            #Region_data = Region_data.applymap(lambda x: [x[i:i+r] if len(x)>=r else x + " "*(r-len(x)) for i in range(len(x)-r+1 if len(x)>=r else 1)] if x else x)
            #CareHome_data = CareHome_data.applymap(lambda x: [x[i:i+r] if len(x)>=r else x + " "*(r-len(x)) for i in range(len(x)-r+1 if len(x)>=r else 1)] if x else x)
            Region_data = [[[p_line[i:i+r] if len(p_line)>=r else p_line + " "*(r-len(p_line)) for i in range(len(p_line)-r+1 if len(p_line)>=r else 1)] if p_line else p_line for p_line in patient] for patient in Region_data]
            CareHome_data = [[[ch_line[i:i+r] if len(ch_line)>=r else ch_line + " "*(r-len(ch_line)) for i in range(len(ch_line)-r+1 if len(ch_line)>=r else 1)] if ch_line else ch_line for ch_line in carehome] for carehome in CareHome_data]
        # Get the non-NaN minimum scores of each Care Home for all address lines (w) to each address line (x)
        #Region_CareHome_scores = Region_data.applymap(lambda x: nan_to_num(np.nanmin([[distance(x, w) for w in list(CareHome[address])] for address in CareHome_data.columns], axis=0), nan=1.0) if x or sum(x) else float("nan"))
        ##Region_CareHome_scores = Region_data.applymap(lambda x: CareHome_data.applymap(lambda w: distance(x,w)).min(axis=1).fillna(value=1.0).to_numpy() if x or sum(x) else float("nan"))
        # Average over non-NaN address lines' values
        #Region_CareHome_scores = Region_CareHome_scores.sum(axis=1)/Region_CareHome_scores.count(axis=1)
        if split_numbers:
            Region_CareHome_scores = [[nan_to_num(np.nanmean([np.nanmin([distance(p_line, ch_line) for ch_line in CareHome_data[carehome]]) for p_line in Region_data[patient]]+[1-any([p_line==ch_line for ch_line in CareHome_numbers[carehome]]) for p_line in Region_numbers[patient]]), nan=1.0) for carehome in range(len(CareHome_data))] for patient in range(len(Region_data))]
        else:
            Region_CareHome_scores = [[nan_to_num(np.nanmean([np.nanmin([distance(p_line, ch_line) for ch_line in carehome]) for p_line in patient]), nan=1.0) for carehome in CareHome_data] for patient in Region_data]
        
        # Postprocessing HERE!
        
        Region_CareHome_scores = pd.Series(Region_CareHome_scores)
        if name:
            # Save scores into csv file
            #Region_CareHome_scores.apply(list).to_csv(name + "_scores.csv", index=False)
            Region_CareHome_scores.to_csv(name + "_scores.csv", index=False)
    else:
        # Load scores from csv file
        Region_CareHome_scores = pd.read_csv(name + "_scores.csv", header=None, squeeze=True, keep_default_na=False, engine="python").apply(lambda x: literal_eval(x))
    
    Region_CareHome_scores = Region_CareHome_scores.values.tolist()
    # Postprocessing
    if filter_PC:
        Region_PC = Region["current_postcode"]
        CareHome_PC = CareHome["Service_Postcode"]
        Region_PC = Region_PC.values.tolist()
        CareHome_PC = CareHome_PC.values.tolist()
        # Clean text
        #Region_PC = Region_PC.apply(lambda x: clean_str(x, postcode=True))
        #CareHome_PC = CareHome_PC.apply(lambda x: clean_str(x, postcode=True))
        Region_PC = [clean_str(p_pc, postcode=True) for p_pc in Region_PC]
        CareHome_PC = [clean_str(ch_pc, postcode=True) for ch_pc in CareHome_PC]
        
        # Check whether Postcode are the same
        consider = False # consider bad Postcodes?
        UK_pc = r"^([Gg][Ii][Rr] ?0[Aa]{2})|((([A-Za-z][0-9]{1,2})|(([A-Za-z][A-Ha-hJ-Yj-y][0-9]{1,2})|(([AZa-z][0-9][A-Za-z])|([A-Za-z][A-Ha-hJ-Yj-y][0-9]?[A-Za-z])))) ?[0-9][A-Za-z]{2})$" # UK Postcode Regular Expression supplied by the UK Government in https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/488478/Bulk_Data_Transfer_-_additional_validation_valid_from_12_November_2015.pdf
        #Region_postcode = Region_PC.apply(lambda x: [pc==x if re.fullmatch(UK_pc, pc) else consider for pc in list(CareHome_PC)] if re.fullmatch(UK_pc, x) else [consider]*len(CareHome_PC))
        Region_postcode = [[carehome==patient if re.search(UK_pc, carehome) else consider for carehome in CareHome_PC] if re.search(UK_pc, patient) else [consider]*len(CareHome_PC) for patient in Region_PC]
        # Match with the same Postcodes
        #Region_CareHome_scores = Region_CareHome_scores.mul(Region_postcode).add(Region_postcode.apply(lambda x: np.invert(x)))
        Region_CareHome_scores = np.add(np.multiply(Region_CareHome_scores,Region_postcode),np.logical_not(Region_postcode)).tolist()
        
    if service_town:
        Region_town = Region.agg("".join, axis=1)
        CareHome_town = CareHome["Service_town"]
        Region_town = Region_town.values.tolist()
        CareHome_town = CareHome_town.values.tolist()
        # Clean text
        #Region_town = Region_town.applymap(lambda x: clean_str(x))
        #CareHome_town = CareHome_town.applymap(lambda x: clean_str(x))
        Region_town = [clean_str(p_town) for p_town in Region_town]
        CareHome_town = [clean_str(ch_town) for ch_town in CareHome_town]
        
        #Region_town = Region_town.apply(lambda x: [True if re.search(y,x) else False for y in CareHome_town])
        ##Region_town = Region_town.apply(lambda x: CareHome_town.apply(lambda y: True if re.search(y,x) else False).to_numpy())
        Region_town = [[True if re.search(carehome,patient) else False for carehome in CareHome_town] for patient in Region_town]
        # Match with the same Service Town
        Region_CareHome_scores = np.add(np.multiply(Region_CareHome_scores,Region_town),np.logical_not(Region_town)).tolist()
        
    if carehome_service:
        # Filter the Older People and Physical and Sensory Impairment subtypes
        services = ["Older People", "Physical and Sensory Impairment"]
        CareHome_service = CareHome["Subtype"].isin(services)
        CareHome_service = CareHome_service.values.tolist()
        # Match with the 
        #Region_CareHome_scores = Region_CareHome_scores.mul(Region_postcode).add(Region_postcode.apply(lambda x: np.invert(x)))
        Region_CareHome_scores = np.add(np.multiply(Region_CareHome_scores,CareHome_service),np.logical_not(CareHome_service)).tolist()
        
    if carehome_name:
        Region_carehome = Region["current_address_L1"]
        Region_carehome = Region_carehome.values.tolist()
        # Clean text
        #Region_carehome = Region_carehome.applymap(lambda x: clean_str(x))
        Region_carehome = [clean_str(p_ch) for p_ch in Region_carehome]
        
        keywords = [
                "carehome", "nursinghome", "residentialhome", "reshome",
                "care home", "nursing home","residential home", "res home",
                "c home", "n home", "r home",
                "nursing carehome", "residential carehome", "res carehome",
                "n carehome", "r carehome",
                "nursing care home", "residential care home", "res care home",
                "n care home", "r care home",
                "care hm","nursing hm","residential hm", "res hm",
                "c hm", "n hm", "r hm",
                "nursing care hm", "residential care hm", "res care hm",
                "n care hm", "r care hm",
                ]
        regex = "|".join([r"\b"+keyword.lower()+r"\b" for keyword in keywords])
        #Region_carehome = Region_carehome.apply(lambda x: [True if re.search(regex,x) else False]*len(CareHome))
        Region_carehome = [[True if re.search(regex,patient) else False]*len(CareHome) for patient in Region_carehome]
        # Classify carehomes with a specific keyword
        Region_CareHome_scores = np.multiply(Region_CareHome_scores,np.logical_not(Region_carehome)).tolist()
        
    Region_CareHome_scores = pd.Series(Region_CareHome_scores)
    return Region_CareHome_scores

def address_matching_fit(Region_CareHome_scores_train, Region_labels_train, Region, CareHome, name=""):
    """ Fit using thes bet scores between a Region list of patients and a Care Home list of addresses."""
    # Get the best matching Care Home address
    #Region_match_CareHome = Region_CareHome_scores_train.apply(lambda x: [np.argmin(x), np.min(x)])
    Region_match_CareHome = [np.argmin(Region_CareHome_scores_train.values.tolist(), axis=1), np.min(Region_CareHome_scores_train.values.tolist(), axis=1)]
    # Search algorithm for the best Performance
    steps = 1000 # Number of thresholds values
    thresholds = np.arange(0, 1+1/steps, 1/steps) # Thresholds values
    #labels = [Region_match_CareHome.str[1]<=threshold for threshold in thresholds]
    labels = [Region_match_CareHome[1]<=threshold for threshold in thresholds]
    # Performance measures
    precision, recall, f1, _ = map(list, zip(*[precision_recall_fscore_support(Region_labels_train, label, average="binary") for label in labels]))
    # Best threshold according to a criterion (Precision / Recall / F1-measure)
    criteria = f1
    argmax_criteria = np.argmax(criteria)
    max_criteria = criteria[argmax_criteria]
    threshold = thresholds[argmax_criteria]
    # Final labels with the best threshold
    labels = labels[argmax_criteria]
    if name:
        print("Best performance (" + name + "): " + str(round(max_criteria*100,2)) + "% with threshold = " + str(threshold))
        # Performance vs Threshold
        plot_thresholds(precision, recall, f1, thresholds, threshold, max_criteria, name + "_prediction")
        # Scores with the best Threshold value
        #plot_scores(Region_match_CareHome.str[1], Region_labels_train, labels, threshold, name + "_prediction")
        plot_scores(Region_match_CareHome[1], Region_labels_train, labels, threshold, name + "_prediction")
        # Print results
        #print_results(Region_labels_train, labels, 1-Region_match_CareHome.str[1], name + "_prediction")
        print_results(Region_labels_train, labels, 1-Region_match_CareHome[1], name + "_prediction")
        # Save scores, labels and best Care Home address match
        df = pd.DataFrame(columns=list(Region.columns) + ["True_Labels", "Predicted_Labels", "Score"] + list(CareHome.columns))
        df[Region.columns] = Region
        df["True_Labels"] = Region_labels_train
        #df["Predicted_Labels"] = labels
        df["Predicted_Labels"] = labels.tolist()
        #df["Score"] = Region_match_CareHome.str[1]
        df["Score"] = Region_match_CareHome[1].tolist()
        if any(labels):
            # Best match Care Home address
            #df.loc[labels, CareHome.columns] = CareHome.iloc[Region_match_CareHome[labels].str[0]].values
            df.loc[labels, CareHome.columns] = CareHome.iloc[Region_match_CareHome[0][labels]].values
        df.to_csv(name + "_prediction.csv", index=False)
    return labels, threshold

def address_matching_predict(Region_CareHome_scores_test, Region_labels_test, Region, CareHome, threshold, name=""):
    """ Predict using the minimum scores between a Region list of patients and a Care Home list of addresses."""
    # Get the best matching Care Home address
    #Region_match_CareHome = Region_CareHome_scores_test.apply(lambda x: [np.argmin(x), np.min(x)])
    Region_match_CareHome = [np.argmin(Region_CareHome_scores_test.values.tolist(), axis=1), np.min(Region_CareHome_scores_test.values.tolist(), axis=1)]
    # Final labels with the best threshold
    #labels = Region_match_CareHome.str[1]<=threshold
    labels = Region_match_CareHome[1]<=threshold
    if name:
        # Print results
        #print_results(Region_labels_test, labels, 1-Region_match_CareHome.str[1], name + "_prediction")
        print_results(Region_labels_test, labels, 1-Region_match_CareHome[1], name + "_prediction")
        # Save scores, labels and best Care Home address match
        df = pd.DataFrame(columns=list(Region.columns) + ["True_Labels", "Predicted_Labels", "Score"] + list(CareHome.columns))
        df[Region.columns] = Region
        df["True_Labels"] = Region_labels_test
        #df["Predicted_Labels"] = labels
        df["Predicted_Labels"] = labels.tolist()
        #df["Score"] = Region_match_CareHome.str[1]
        df["Score"] = Region_match_CareHome[1].tolist()
        if any(labels):
            # Best match Care Home address
            #df.loc[labels, CareHome.columns] = CareHome.iloc[Region_match_CareHome[labels].str[0]].values
            df.loc[labels, CareHome.columns] = CareHome.iloc[Region_match_CareHome[0][labels]].values
        df.to_csv(name + "_prediction.csv", index=False)
    return labels
