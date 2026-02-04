---
title: "Understanding Gradient Descent in Deep Learning"
date: 2025-01-12
categories: [Deep Learning]
tags: [deep learning]
math: true
---

### 1. Mathematical Definition
* **Derivative Coefficient**: This refers to the numerical value representing the **rate of change** of a function's value at a specific point, or the **slope of the tangent line** at that point.
* In Deep Learning, calculating this value is an absolute requirement for executing **Gradient Descent**.

### 2. Derivative = 0 vs. Gradient Descent
Why do we use Gradient Descent instead of simply finding the point where the derivative is zero?

* **Mathematical Ideal (Closed-form Solution)**: For simple functions like a quadratic equation, we can solve $$f'(x) = 0$$ algebraically to find the minimum in one step.
* **The Reality of Deep Learning**:
    * **No Closed-form**: Most loss functions in DL are non-convex and cannot be solved with a simple formula.
    * **High Complexity**: Loss functions involve millions of parameters ($$W$$, $$b$$), making it computationally impossible to solve for zero simultaneously.
    * **Efficiency**: When dealing with massive datasets, Gradient Descent is far more computationally efficient.

### 3. The Logic of Gradient Descent
The direction of the update depends on the sign of the derivative. The update rule is: 

$$W_{new} = W_{old} - \alpha \cdot \frac{\partial Loss}{\partial W}$$


* **If the derivative is positive (+)**:
    * The function value is increasing as $$W$$ increases.
    * **Action**: Move in the **opposite direction** (decrease $$W$$) to find the minimum.
* **If the derivative is negative (-)**:
    * The function value is decreasing as $$W$$ increases.
    * **Action**: **Continue in that direction** (increase $$W$$) to reach the minimum.

### 4. The Essence of Training
Ultimately, training a deep learning model is the continuous process of updating numerous **weights ($$W$$)** by calculating the derivatives of the loss function until the gradient reaches near zero (the minimum point).