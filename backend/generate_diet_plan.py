def generate_diet_plan(stage, age, gender):
    """
    Returns a simple nutrient plan suggestion based on cancer stage.
    """
    diet_plans = {
        "Stage I": {
            "Morning": "Oatmeal with fruits and green tea",
            "Lunch": "Grilled vegetables and lentil soup",
            "Dinner": "Light salad with olive oil and tofu"
        },
        "Stage II": {
            "Morning": "Smoothie with spinach, banana, and chia seeds",
            "Lunch": "Brown rice, steamed broccoli, and beans",
            "Dinner": "Vegetable soup and whole grain bread"
        },
        "Stage III": {
            "Morning": "Green tea and boiled eggs",
            "Lunch": "Steamed fish, veggies, and brown rice",
            "Dinner": "Soup and low-fat yogurt"
        },
        "Stage IV": {
            "Morning": "Protein shake and fruit bowl",
            "Lunch": "High-calorie foods like nuts, avocados, and whole grains",
            "Dinner": "Vegetable stew and soft-cooked grains"
        }
    }

    plan = diet_plans.get(stage, {
        "Morning": "Fruits and hydration-focused breakfast",
        "Lunch": "Balanced meal with proteins and fibers",
        "Dinner": "Light and nutrient-rich foods"
    })

    return plan
