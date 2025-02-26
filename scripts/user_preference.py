import json
import os

# ----------------------------------------------
# Configuration
# ----------------------------------------------
PREFERENCE_FILE = "user_preferences.json"

DEFAULT_PREFERENCES = {
    "math_format": "LaTeX",
    "reasoning_style": "Step-by-step with explanations",
    "conciseness": "Medium",
    "memory_persistence": "High",
}

# ----------------------------------------------
# Load & Save Preferences
# ----------------------------------------------
def load_user_preferences():
    """
    Loads user preferences from a JSON file. If not found, loads defaults.

    Returns:
        dict: User preferences.
    """
    if os.path.exists(PREFERENCE_FILE):
        with open(PREFERENCE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        save_user_preferences(DEFAULT_PREFERENCES)
        return DEFAULT_PREFERENCES

def save_user_preferences(preferences):
    """
    Saves user preferences to a JSON file.

    Args:
        preferences (dict): Dictionary of user preferences.

    Returns:
        None
    """
    with open(PREFERENCE_FILE, "w", encoding="utf-8") as f:
        json.dump(preferences, f, indent=4)
    print("User preferences updated.")

# ----------------------------------------------
# Modify Preferences
# ----------------------------------------------
def update_user_preference(key, value):
    """
    Updates a specific user preference.

    Args:
        key (str): Preference key.
        value (str/int): New value.

    Returns:
        None
    """
    preferences = load_user_preferences()
    if key in preferences:
        preferences[key] = value
        save_user_preferences(preferences)
        print(f"Updated '{key}' to '{value}'.")
    else:
        print(f"Invalid preference key: {key}")

# ----------------------------------------------
# Display Preferences
# ----------------------------------------------
def display_user_preferences():
    """
    Prints the current user preferences.
    """
    preferences = load_user_preferences()
    print(json.dumps(preferences, indent=4))

# ----------------------------------------------
# Example Usage
# ----------------------------------------------
if __name__ == "__main__":
    display_user_preferences()
    update_user_preference("reasoning_style", "Detailed explanations with bullet points")
    display_user_preferences()
