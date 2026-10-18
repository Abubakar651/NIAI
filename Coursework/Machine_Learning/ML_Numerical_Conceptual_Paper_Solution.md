# MACHINE LEARNING – NUMERICAL / CONCEPTUAL PAPER

**Student Name:** Abubakar Iqbal  
**Program:** NAVTTC / NIAI (NETSOL Technologies)  
**Time Allowed:** 75 Minutes  
**Total Marks:** 50  

---

### Instructions
- Attempt all questions.
- Show all working for numerical questions.
- Calculations performed manually without ML libraries.
- Probabilities rounded to 3 decimal places where needed.
- $e \approx 2.718$ used where required.

---

## SECTION A — SIMPLE LINEAR REGRESSION [7 Marks]

**Q1. A shop records advertising expenditure $X$ (thousand PKR) and weekly sales $Y$ (thousand PKR).**

| Advertising $X$ | Sales $Y$ |
| :---: | :---: |
| 1 | 10 |
| 2 | 14 |
| 3 | 18 |
| 4 | 22 |

### a) Form of Simple Linear Regression Equation
$$\hat{y} = b_0 + b_1 x$$

### b) Estimating $b_1$ and $b_0$ using first and last observations
- First observation: $(x_1, y_1) = (1, 10)$
- Last observation: $(x_4, y_4) = (4, 22)$

**Step 1: Estimate slope ($b_1$):**
$$b_1 = \frac{y_4 - y_1}{x_4 - x_1} = \frac{22 - 10}{4 - 1} = \frac{12}{3} = 4$$

**Step 2: Estimate intercept ($b_0$):**
Substitute $(x_1, y_1) = (1, 10)$ and $b_1 = 4$:
$$10 = b_0 + 4(1)$$
$$b_0 = 10 - 4 = 6$$

Fitted equation:
$$\hat{y} = 6 + 4x$$

### c) Predict weekly sales when $X = 5$
$$\hat{y} = 6 + 4(5) = 6 + 20 = 26 \text{ thousand PKR (PKR 26,000)}$$

---

## SECTION B — MULTIPLE LINEAR REGRESSION [5 Marks]

**Q2. A house-price model is already trained as:**
$$\text{Price (million PKR)} = 2 + 0.5(\text{Area in 100 sq. yd.}) + 1.0(\text{Bedrooms})$$

### a) Predict the price when $\text{Area} = 10$ and $\text{Bedrooms} = 4$
$$\text{Price} = 2 + 0.5(10) + 1.0(4) = 2 + 5.0 + 4.0 = 11.0 \text{ million PKR}$$

### b) Meaning of coefficient $0.5$
The coefficient $0.5$ indicates that for every 1 unit increase in Area (i.e., every additional 100 sq. yd.), the predicted house price increases by 0.5 million PKR (PKR 500,000), holding the number of bedrooms constant.

### c) Predicted price change if Bedrooms increases by 1 (Area unchanged)
Since the coefficient for Bedrooms is $1.0$, the predicted price will increase by **1.0 million PKR (PKR 1,000,000)**.

---

## SECTION C — BINARY LOGISTIC REGRESSION [8 Marks]

**Q3. A bank predicts default ($1 = \text{Default}, 0 = \text{No Default}$):**
$$z = -4 + 0.05(\text{Age}) + 0.04(\text{Debt Ratio} \times 10)$$
$$P(Y=1) = \frac{1}{1 + e^{-z}}$$

### a) Calculate $z$ for $\text{Age} = 40$ and $\text{Debt Ratio} \times 10 = 20$
$$z = -4 + 0.05(40) + 0.04(20)$$
$$z = -4 + 2.0 + 0.8 = -1.2$$

### b) Approximate probability of default (using $e^{1.2} \approx 3.32$)
$$P(Y=1) = \frac{1}{1 + e^{-(-1.2)}} = \frac{1}{1 + e^{1.2}} = \frac{1}{1 + 3.32} = \frac{1}{4.32} \approx 0.231 \text{ (or } 23.1\%)$$

### c) Classify customer using threshold $0.50$
Since $P(Y=1) = 0.231 < 0.50$, the customer is classified as **0 (No Default)**.

### d) Does prediction change if threshold changes to $0.70$?
No. Because $0.231 < 0.70$, the classification remains **0 (No Default)**.

---

## SECTION D — MULTINOMIAL LOGISTIC REGRESSION [10 Marks]

**Q4. Career prediction model: $0 = \text{Data Scientist}, 1 = \text{Data Analyst}, 2 = \text{Software Engineer}$.**
Scores for student: $z_0 = 2, z_1 = 1, z_2 = 0$.

### a) Exponent calculation ($e^2 \approx 7.39, e^1 \approx 2.72, e^0 = 1$)
$$\sum_{j=0}^2 e^{z_j} = 7.39 + 2.72 + 1.00 = 11.11$$

### b) Softmax probabilities: $P(Y=k) = \frac{e^{z_k}}{\sum e^{z_j}}$
- **Data Scientist ($k=0$):**
  $$P(Y=0) = \frac{7.39}{11.11} \approx 0.665 \text{ (66.5\%)}$$
- **Data Analyst ($k=1$):**
  $$P(Y=1) = \frac{2.72}{11.11} \approx 0.245 \text{ (24.5\%)}$$
- **Software Engineer ($k=2$):**
  $$P(Y=2) = \frac{1.00}{11.11} \approx 0.090 \text{ (9.0\%)}$$

*(Verification sum: $0.665 + 0.245 + 0.090 = 1.000$)*

### c) Predicted career
**Data Scientist** is predicted because it has the highest probability ($0.665$).

### d) Why is the denominator needed in Softmax?
The denominator normalizes the exponential values so that the resulting scores represent a true probability distribution: every probability lies between 0 and 1, and the sum of probabilities across all mutually exclusive classes equals exactly 1.

---

## SECTION E — K-NEAREST NEIGHBORS (KNN) [10 Marks]

**Q5. New student: $\text{Study Hours} = 5, \text{Attendance} = 80$. Use Euclidean distance and $K = 3$.**

| Student | Study Hours | Attendance | Result |
| :---: | :---: | :---: | :---: |
| A | 4 | 70 | Fail |
| B | 6 | 75 | Pass |
| C | 5 | 85 | Pass |
| D | 8 | 90 | Pass |
| E | 3 | 60 | Fail |

### a) Calculate Euclidean distances: $d = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2}$
- **To Student A $(4, 70)$:**
  $$d_A = \sqrt{(5-4)^2 + (80-70)^2} = \sqrt{1^2 + 10^2} = \sqrt{1 + 100} = \sqrt{101} \approx 10.050$$
- **To Student B $(6, 75)$:**
  $$d_B = \sqrt{(5-6)^2 + (80-75)^2} = \sqrt{(-1)^2 + 5^2} = \sqrt{1 + 25} = \sqrt{26} \approx 5.099$$
- **To Student C $(5, 85)$:**
  $$d_C = \sqrt{(5-5)^2 + (80-85)^2} = \sqrt{0^2 + (-5)^2} = \sqrt{0 + 25} = \sqrt{25} = 5.000$$
- **To Student D $(8, 90)$:**
  $$d_D = \sqrt{(5-8)^2 + (80-90)^2} = \sqrt{(-3)^2 + (-10)^2} = \sqrt{9 + 100} = \sqrt{109} \approx 10.440$$
- **To Student E $(3, 60)$:**
  $$d_E = \sqrt{(5-3)^2 + (80-60)^2} = \sqrt{2^2 + 20^2} = \sqrt{4 + 400} = \sqrt{404} \approx 20.099$$

### b) Identify 3 nearest neighbors ($K = 3$)
Ranking distances from smallest to largest:
1. Student C ($d = 5.000$)
2. Student B ($d = 5.099$)
3. Student A ($d = 10.050$)

The 3 nearest neighbors are **C, B, and A**.

### c) Majority voting prediction
- Student C: Pass
- Student B: Pass
- Student A: Fail

Votes: 2 Pass vs 1 Fail.  
Predicted result: **Pass**.

### d) Why feature scaling is important for KNN
KNN relies on geometric distance. Features with large ranges (e.g. Attendance ranging 60–90) disproportionately dominate the Euclidean distance calculation compared to features with small ranges (e.g. Study Hours ranging 3–8). Feature scaling normalizes all dimensions to equal footing.

---

## SECTION F — NAIVE BAYES [10 Marks]

**Q6. Dataset with 10 emails (Spam = 4, Not Spam = 6). Word 'Offer' count:**

| Class | Total Emails | Emails containing 'Offer' |
| :---: | :---: | :---: |
| Spam | 4 | 3 |
| Not Spam | 6 | 1 |

### a) Calculate prior probabilities
$$P(\text{Spam}) = \frac{4}{10} = 0.400$$
$$P(\text{Not Spam}) = \frac{6}{10} = 0.600$$

### b) Calculate likelihoods
$$P(\text{Offer} \mid \text{Spam}) = \frac{3}{4} = 0.750$$
$$P(\text{Offer} \mid \text{Not Spam}) = \frac{1}{6} \approx 0.167$$

### c) Calculate classification scores
$$\text{Score}(\text{Spam}) = P(\text{Spam}) \times P(\text{Offer} \mid \text{Spam}) = 0.400 \times 0.750 = 0.300$$
$$\text{Score}(\text{Not Spam}) = P(\text{Not Spam}) \times P(\text{Offer} \mid \text{Not Spam}) = 0.600 \times \frac{1}{6} = 0.100$$

### d) Prediction for email containing 'Offer'
Since $\text{Score}(\text{Spam}) = 0.300 > \text{Score}(\text{Not Spam}) = 0.100$, Naive Bayes will predict **Spam**.

### e) Why is it called 'Naive' Bayes?
It is termed "Naive" because it makes the simplifying assumption that all predictor features are conditionally independent of each other given the class label, which is rarely strictly true in real-world data.
