package com.android.victor.model

object Data {

    fun generateDataCategory(): ArrayList<MessageModel>{
        val categories: ArrayList<MessageModel> = ArrayList()
        categories.add(MessageModel("Body", "victor", false, "body"))
        categories.add(MessageModel("Digestion", "victor", false, "digestion"))
        categories.add(MessageModel("Eyes", "victor", false, "eyes"))
        categories.add(MessageModel("Mental", "victor", false, "mental"))
        categories.add(MessageModel("Sensation", "victor", false, "sensation"))
        categories.add(MessageModel("Skin", "victor", false, "skin"))
        categories.add(MessageModel("THT", "victor", false, "tht"))
        categories.add(MessageModel("Additional", "victor", false, "additional"))
        categories.add(MessageModel("Finish", "victor", false, "finish"))
        return categories
    }

    fun generateDataBody(): ArrayList<MessageModel>{
        val bodies: ArrayList<MessageModel> = ArrayList()
        bodies.add(MessageModel("Muscle Wasting", messageId = "muscle_wasting"))
        bodies.add(MessageModel("Weakness in Limbs", messageId = "weakness_in_limbs"))
        bodies.add(MessageModel("Weakness of One Body Side", messageId = "weakness_of_one_body_side"))
        bodies.add(MessageModel("Scurring", messageId = "scurring"))
        bodies.add(MessageModel("Swollen Blood Vessels", messageId = "swollen_blood_vessels"))
        bodies.add(MessageModel("Cramps", messageId = "cramps"))
        bodies.add(MessageModel("Knee Pain", messageId = "knee_pain"))
        bodies.add(MessageModel("Movement Stiffness", messageId = "movement_stiffness"))
        bodies.add(MessageModel("Hip Joint Pain", messageId = "hip_joint_pain"))
        bodies.add(MessageModel("Swollen Legs", messageId = "swollen_legs"))
        bodies.add(MessageModel("Prominent Veins on Calf", messageId = "prominent_veins_on_calf"))
        bodies.add(MessageModel("Belly Pain", messageId = "belly_pain"))
        bodies.add(MessageModel("Stiff Neck", messageId = "stiff_neck"))
        bodies.add(MessageModel("Back Pain", messageId = "back_pain"))
        bodies.add(MessageModel("Constipation", messageId = "constipation"))
        bodies.add(MessageModel("Neck Pain", messageId = "neck_pain"))
        bodies.add(MessageModel("Swelling Joints", messageId = "swelling_joints"))
        bodies.add(MessageModel("Muscle Weakness", messageId = "muscle_weakness"))
        bodies.add(MessageModel("Muscle Pain", messageId = "muscle_pain"))
        bodies.add(MessageModel("Irritability", messageId = "irritability"))
        bodies.add(MessageModel("Joint Pain", messageId = "joint_pain"))
        bodies.add(MessageModel("Chest Pain", messageId = "chest_pain"))
        bodies.add(MessageModel("Abdominal Pain", messageId = "abdominal_pain"))
        bodies.add(MessageModel("Fatigue", messageId = "fatigue"))
        bodies.add(MessageModel("Back", messageId = "back"))
        return bodies
    }

    fun generateDataDigestion(): ArrayList<MessageModel>{
        val digestion: ArrayList<MessageModel> = ArrayList()
        digestion.add(MessageModel("Dehydration", messageId = "dehydration"))
        digestion.add(MessageModel("Spotting Urination", messageId = "spotting_urination"))
        digestion.add(MessageModel("Pain During Bowel Movements", messageId = "pain_during_bowel_movements"))
        digestion.add(MessageModel("Weight Gain", messageId = "weight_gain"))
        digestion.add(MessageModel("Bladder Biscomfort", messageId = "bladder_discomfort"))
        digestion.add(MessageModel("Pain in Anal Region", messageId = "pain_in_anal_region"))
        digestion.add(MessageModel("Swelling of Stomach", messageId = "swelling_of_stomach"))
        digestion.add(MessageModel("Bloody Stool", messageId = "bloody_stool"))
        digestion.add(MessageModel("Continuous Feel of Urine", messageId = "continuous_feel_of_urine"))
        digestion.add(MessageModel("Distention of Abdomen", messageId = "distention_of_abdomen"))
        digestion.add(MessageModel("Irregular Sugar Level", messageId = "irregular_sugar_level"))
        digestion.add(MessageModel("Irritation in Anus", messageId = "irritation_in_anus"))
        digestion.add(MessageModel("Passage of Gases", messageId = "passage_of_gases"))
        digestion.add(MessageModel("Yellow Urine", messageId = "yellow_urine"))
        digestion.add(MessageModel("Toxic Took (typhos)", messageId = "toxic_look_(typhos)"))
        digestion.add(MessageModel("Acute Liver Failure", messageId = "acute_liver_failure"))
        digestion.add(MessageModel("Enlarged Thyroid", messageId = "enlarged_thyroid"))
        digestion.add(MessageModel("Increased Appetite", messageId = "increased_appetite"))
        digestion.add(MessageModel("Polyuria", messageId = "polyuria"))
        digestion.add(MessageModel("Stomach Bleeding", messageId = "stomach_bleeding"))
        digestion.add(MessageModel("Congestion", messageId = "congestion"))
        digestion.add(MessageModel("Burning Micturition", messageId = "burning_micturition"))
        digestion.add(MessageModel("Stomach Pain", messageId = "stomach_pain"))
        digestion.add(MessageModel("Acidity", messageId = "acidity"))
        digestion.add(MessageModel("Indigestion", messageId = "indigestion"))
        digestion.add(MessageModel("Obesity", messageId = "obesity"))
        digestion.add(MessageModel("Weight Loss", messageId = "weight_loss"))
        digestion.add(MessageModel("Diarrhoea", messageId = "diarrhoea"))
        digestion.add(MessageModel("Dark Urine", messageId = "dark_urine"))
        digestion.add(MessageModel("Vomiting", messageId = "vomiting"))
        digestion.add(MessageModel("Back", messageId = "back"))
        return digestion
    }

    fun generateDataEyes(): ArrayList<MessageModel> {
        val eyes: ArrayList<MessageModel> = ArrayList()
        eyes.add(MessageModel("Sunken Eyes", messageId = "sunken_eyes"))
        eyes.add(MessageModel("Watering From Eyes", messageId = "watering_from_eyes"))
        eyes.add(MessageModel("Puffy Face And Eyes", messageId="puffy_face_and_eyes"))
        eyes.add(MessageModel("Pain Behind The Eyes", messageId = "pain_behind_the_eyes"))
        eyes.add(MessageModel("Redness of Eyes", messageId = "redness_of_eyes"))
        eyes.add(MessageModel("Blurred and Distorted Vision", messageId = "blurred_and_distorted_vision"))
        eyes.add(MessageModel("Yellowing of Eyes", messageId = "yellowing_of_eyes"))
        eyes.add(MessageModel("Back", messageId = "back"))
        return eyes
    }

    fun generateDataMental(): ArrayList<MessageModel> {
        val mental: ArrayList<MessageModel> = ArrayList()
        mental.add(MessageModel("Anxiexty", messageId = "Anxiexty"))
        mental.add(MessageModel("Lack Of Concentration", messageId = "lack_of_concentration"))
        mental.add(MessageModel("Visual Disturbances", messageId="visual_disturbances"))
        mental.add(MessageModel("Mood Swings", messageId = "mood_swings"))
        mental.add(MessageModel("Depression", messageId = "depression"))
        mental.add(MessageModel("Lethargy", messageId = "lethargy"))
        mental.add(MessageModel("High Fever", messageId = "high_fever"))
        mental.add(MessageModel("Back", messageId = "back"))
        return mental
    }

    fun generateDataSensation(): ArrayList<MessageModel> {
        val sensation: ArrayList<MessageModel> = ArrayList()
        sensation.add(MessageModel("Shivering'", messageId = "shivering'"))
        sensation.add(MessageModel("Spinning Movements", messageId = "spinning_movements"))
        sensation.add(MessageModel("Cold Hands And Feets", messageId="cold_hands_and_feets"))
        sensation.add(MessageModel("Altered Sensorium", messageId = "altered_sensorium"))
        sensation.add(MessageModel("unsteadiness", messageId = "unsteadiness"))
        sensation.add(MessageModel("Internal Itching", messageId = "internal_itching"))
        sensation.add(MessageModel("Slurred Speech", messageId = "slurred_speech"))
        sensation.add(MessageModel("Coma", messageId = "coma"))
        sensation.add(MessageModel("Palpitations", messageId = "palpitations"))
        sensation.add(MessageModel("Restlessness", messageId = "restlessness"))
        sensation.add(MessageModel("Painful Walking", messageId = "painful_walking"))
        sensation.add(MessageModel("Fast Heart Rate", messageId = "fast_heart_rate"))
        sensation.add(MessageModel("Dizziness", messageId = "dizziness"))
        sensation.add(MessageModel("Loss of Balance", messageId = "loss_of_balance"))
        sensation.add(MessageModel("Excessive Hunger", messageId = "excessive_hunger"))
        sensation.add(MessageModel("Malaise", messageId = "malaise"))
        sensation.add(MessageModel("Chills'", messageId = "chills'"))
        sensation.add(MessageModel("Headache", messageId = "headache"))
        sensation.add(MessageModel("Nausea", messageId = "nausea"))
        sensation.add(MessageModel("Loss of Appetite", messageId = "loss_of_appetite"))
        sensation.add(MessageModel("Back", messageId = "back"))
        return sensation
    }

    fun generateDataSkin(): ArrayList<MessageModel> {
        val skin: ArrayList<MessageModel> = ArrayList()
        skin.add(MessageModel("Foul Smell of Urine", messageId = "foul_smell_of_urine"))
        skin.add(MessageModel("Pus Filled Pimples", messageId = "pus_filled_pimples"))
        skin.add(MessageModel("Blackheads", messageId = "blackheads"))
        skin.add(MessageModel("Nodal Skin Eruptions", messageId = "nodal_skin_eruptions"))
        skin.add(MessageModel("Dischromic Patches", messageId = "dischromic_patches"))
        skin.add(MessageModel("Blister", messageId = "blister"))
        skin.add(MessageModel("Skin Peeling", messageId = "skin_peeling"))
        skin.add(MessageModel("Bruising", messageId = "bruising"))
        skin.add(MessageModel("Red Sore Around Nose", messageId = "red_sore_around_nose"))
        skin.add(MessageModel("Silver Like Dusting", messageId = "silver_like_dusting"))
        skin.add(MessageModel("Yellow Crust Ooze", messageId = "yellow_crust_ooze"))
        skin.add(MessageModel("Small Dents in Nails", messageId = "small_dents_in_nails"))
        skin.add(MessageModel("Inflammatory Nails", messageId = "inflammatory_nails"))
        skin.add(MessageModel("Fluid Overload", messageId = "fluid_overload"))
        skin.add(MessageModel("Brittle Nails", messageId = "brittle_nails"))
        skin.add(MessageModel("Swollen Extremeties", messageId = "swollen_extremeties"))
        skin.add(MessageModel("Red Spot Over Body", messageId = "red_spots_over_body"))
        skin.add(MessageModel("Abnormal Menstruation", messageId = "abnormal_menstruation"))
        skin.add(MessageModel("Swelled Lymph Nodes", messageId = "swelled_lymph_nodes"))
        skin.add(MessageModel("Itching", messageId = "itching"))
        skin.add(MessageModel("Sweating", messageId = "sweating"))
        skin.add(MessageModel("Skin Rash", messageId = "skin_rash"))
        skin.add(MessageModel("Yellowish Skin", messageId = "yellowish_skin"))
        skin.add(MessageModel("Back", messageId = "back"))
        return skin
    }

    fun generateDataTht(): ArrayList<MessageModel> {
        val tht: ArrayList<MessageModel> = ArrayList()
        tht.add(MessageModel("Mucoid Sputum", messageId = "mucoid_sputum"))
        tht.add(MessageModel("Drying and Tingling Lips", messageId = "drying_and_tingling_lips"))
        tht.add(MessageModel("Throat Irritation", messageId="throat_irritation"))
        tht.add(MessageModel("Rusty Sputum", messageId = "rusty_sputum"))
        tht.add(MessageModel("Sinus Pressure", messageId = "sinus_pressure"))
        tht.add(MessageModel("Runny Nose", messageId = "runny_nose"))
        tht.add(MessageModel("Loss of Smell", messageId = "loss_of_smell"))
        tht.add(MessageModel("Continuous Sneezing", messageId = "continuous_sneezing"))
        tht.add(MessageModel("Phlegm", messageId = "phlegm"))
        tht.add(MessageModel("Breathlessness", messageId = "breathlessness"))
        tht.add(MessageModel("Cough", messageId = "cough"))
        tht.add(MessageModel("Patches in Throat", messageId = "patches_in_throat"))
        tht.add(MessageModel("Ulcers on Tongue", messageId = "ulcers_on_tongue"))
        tht.add(MessageModel("Back", messageId = "back"))
        return tht
    }

    fun generateDataAdditional(): ArrayList<MessageModel> {
        val additional: ArrayList<MessageModel> = ArrayList()
        additional.add(MessageModel("Extra Marital Contacts", messageId = "extra_marital_contacts"))
        additional.add(MessageModel("History of Alcohol Consumption", messageId = "history_of_alcohol_consumption"))
        additional.add(MessageModel("Receiving Blood Transfusion", messageId="receiving_blood_transfusion"))
        additional.add(MessageModel("Receiving Unsterile Injections", messageId = "receiving_unsterile_injections"))
        additional.add(MessageModel("Family History", messageId = "family_history"))
        additional.add(MessageModel("Mild Fever", messageId = "mild_fever"))
        additional.add(MessageModel("High Fever", messageId = "high_fever"))
        additional.add(MessageModel("Back", messageId = "back"))
        return additional
    }
}