def generate_advice(stage, age, gender):
    """
    Generates simple textual advice based on stage, age, and gender.
    """
    base_advice = {
        "Stage I": "Maintain regular checkups and healthy habits.",
        "Stage II": "Consult your oncologist for early treatment options.",
        "Stage III": "Stay consistent with your treatment and rest properly.",
        "Stage IV": "Seek specialized care and emotional support."
    }

    general = " Maintain a balanced diet, avoid smoking, and get regular exercise."

    if age > 60:
        general += " Pay extra attention to your immunity and hydration."
    elif age < 30:
        general += " Early detection gives strong recovery chances."

    if gender == "female":
        general += " Regular screenings for women’s health are advised."
    elif gender == "male":
        general += " Stay consistent with physical activity and stress control."

    return base_advice.get(stage, "Consult a doctor for further evaluation.") + general
