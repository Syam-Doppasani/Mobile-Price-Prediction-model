# 📱 Mobile Price Range Prediction

A machine learning-based web application that predicts the price range of a mobile phone based on its specifications. The app uses a trained **MLPClassifier** neural network model and is built using **Streamlit** for the frontend.

---

## 🚀 Features

- 🔋 Takes key mobile specs as input (battery, camera, screen, memory, etc.)
- 🎯 Predicts mobile price range: `Low`, `Medium`, `High`, `Very High`
- 📊 Displays feature importance from the model
- 🖥️ User-friendly Streamlit web interface
- 🎨 Clean dark-themed UI with CSS customization

---

## 🧠 Model

- Model Type: `MLPClassifier` from `scikit-learn`
- Training Method: Supervised Learning
- Target Variable: `price_range` (0 = low, 1 = medium, 2 = high, 3 = very high)
- Preprocessing: StandardScaler for feature scaling

---



# 📋 Dataset Columns
```
Column Name	Type	Description
battery_power	int	Battery capacity in mAh
blue	int	Bluetooth support (1 = Yes, 0 = No)
clock_speed	float	Processor speed in GHz
dual_sim	int	Dual SIM support (1 = Yes, 0 = No)
fc	int	Front camera megapixels
four_g	int	4G support (1 = Yes, 0 = No)
int_memory	int	Internal storage in GB
m_deep	float	Mobile depth (thickness) in cm
mobile_wt	int	Weight of the phone in grams
n_cores	int	Number of processor cores
pc	int	Primary camera megapixels
px_height	int	Screen resolution height in pixels
px_width	int	Screen resolution width in pixels
ram	int	RAM in MB
sc_h	int	Screen height in cm
sc_w	int	Screen width in cm
talk_time	int	Battery backup in hours under talk usage
three_g	int	3G support (1 = Yes, 0 = No)
touch_screen	int	Touchscreen support (1 = Yes, 0 = No)
wifi	int	WiFi support (1 = Yes, 0 = No)
price_range	int	Target variable:

                             `0` = Low cost  
                             `1` = Medium cost  
                             `2` = High cost  
                             `3` = Very high cost |
```
# 📊 Preview of model
![Screenshot 2025-06-10 232334](https://github.com/user-attachments/assets/cedc4242-2a90-471c-8fb5-ce67ca652f7b)
![Screenshot 2025-06-10 232223](https://github.com/user-attachments/assets/e682b499-caf0-49b9-b8ae-3d17905ef3d5)

# 👨‍💻 Author
Syam Doppasani

For freelance work or collaborations: syamdoppasani@gmail.com
