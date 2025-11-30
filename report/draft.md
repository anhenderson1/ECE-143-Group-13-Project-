## Our Tasks
1. Perform exploratory data analysis (EDA) and visualizations
> Finished. You can find the results in the [Notebook](../notebook/lifestyle_datamining.ipynb). There is an important finding:
> * The dataset is extremely balanced. While this may not perfectly reflect the distribution of the real-world population, it provides an ideal environment for our correlation analysis because it eliminates statistical biases that may arise due to differences in sample size.

2. Analyze correlations between lifestyle variables and health/fitness metrics
> Finished. We have several findings and Interesting facts：
> * A person's BMI or body fat percentage is unrelated to the calories they burn during exercise.
> * An individual's dietary habits and their exercise habits are disconnected.
> * Strength training burns more calories than cardio exercise like running. This **contradicts our assumption**, because cardio should be more tiring, increases heart rate, and burns fat faster.
> * It's clear that as water intake increases, mood ratings improve. Thus, drinking plenty of water regularly is not only good for your health, but it can also improve your mood.
> * If you choose the Paleo or Keto diet, you'll feel better than if you follow a balanced diet.
> * As average workout heart rate (avg_bpm) increases, the mood rating tends to decrease slightly. In other words, higher workout intensity is associated with lower post-exercise mood.

3. Train a Random Forest classifier to predict calorie burn categories: Low, Medium, High, Very High
> Finished. With the RF classifier and the `SHAP` visualization tool, we have some findings:
> * Our model primarily relies on exercise-related behavioral features — particularly session duration, workout type, expected burn, and calorie balance — to make predictions.
> * It has been proven that gender has no impact on calorie consumption regardless of gender. Women are not at a physiological disadvantage, nor are men at a physiological advantage; as long as you exercise, there is no difference in calorie consumption.

4. Interpret feature importance to identify which factors matter most
Based on the Random Forest and SHAP analysis, we found that calorie burn is primarily determined by workout behavior rather than demographic or physiological traits.

**Key Factors That Matter Most**

* Session Duration
* * The strongest predictor. Longer workouts consistently lead to higher calorie burn. 
* Workout Type (HIIT, Strength, Cardio, Yoga)
* * HIIT and Strength strongly contribute to High / Very High burn.
* * Cardio is more related to Medium burn.
* Expected Burn & Calorie Balance
* * Reflect user intention and energy availability; both significantly influence outcomes.
* Experience Level & Workout Frequency
* * More consistent and experienced exercisers show more predictable calorie burn patterns.

**Factors That Do Not Matter**

* Gender, BMI, body-fat %, height, weight
* SHAP confirms these variables have near-zero influence on burn category.

**Overall Insight**

> Calorie burn depends on what you do—not who you are. 
> Behavioral choices (duration, intensity, consistency) dominate over biology.

* Practical Health Advice

* * Focus on longer sessions, higher intensity, and training consistency.
* * Don’t overestimate gender or body type differences—anyone can increase calorie burn with effective workout behaviors.