import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

env = gym.make("CartPole-v1")

def run_episode():
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = env.action_space.sample()
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    return state, total_reward

data = []
NUM_EPISODES = 1000

print(f"Generating data from {NUM_EPISODES} episodes...")

for _ in range(NUM_EPISODES):
    final_state, episode_reward = run_episode()
    data.append([
        final_state[0],
        final_state[1],
        final_state[2],
        final_state[3],
        episode_reward
    ])

columns = [
    "cart_position",
    "cart_velocity",
    "pole_angle",
    "pole_angular_velocity",
    "episode_reward"
]

df = pd.DataFrame(data, columns=columns)
print("Data Generation Complete. Sample data:")
print(df.head())

X = df.drop("episode_reward", axis=1)
y = df["episode_reward"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "KNN": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append([name, mae, rmse, r2])

results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "RMSE", "R2 Score"]
)

print("\nModel Comparison (Sorted by RMSE):")
print(results_df.sort_values("RMSE"))

best_model = RandomForestRegressor(random_state=42)
best_model.fit(X_train, y_train)
best_preds = best_model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, best_preds, alpha=0.5, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Episode Reward")
plt.ylabel("Predicted Episode Reward")
plt.title("Actual vs Predicted Episode Reward (Random Forest)")
plt.grid(True)
plt.show()

env.close()
