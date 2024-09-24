# Problems

## Question 1

The only correct statement is C. The justification is:

- The variable $x_1$ represents the time of day in blocks of 30 minutes
  that partition a day into a finite number of intervals. The variable is
  therefore discrete. Since the intervals can be ordered, the variable is
  ordinal.
- The attribute $x_6$ is a ratio variable. The number of broken traffic
  lights is clearly a numeric variable. The cannonical zero of the variable
  is zero broken traffic lights.
- The attribute $x_7$ is a ratio variable for much the same reasons as for
  $x_6$.
- The congestion level is ordinal as it is a discrete variable that can be
  ordered.

The statements A, B, and D are all incorrect as they mistake $x_1$ for
something other than an ordinal variable.

## Question 2

- This statement is correct since $|26 - 19| = 7$ and that is the maximal
  difference among the respective coordinates of the two vectors.
- The metric is
  $$d_3(\mathbf{x}_{14}, \mathbf{x}_{18}) = \sqrt[3]{|26 - 19|^3 + |2 - 0|^3} \approx 7.05.$$
  Thus, the statement is incorrect.
- The metric is
  $$d_1(\mathbf{x}_{14}, \mathbf{x}_{18}) = |26 - 19| + |2 - 0| = 9.$$
  Thus, the statement is incorrect.
- The metric is
  $$d_4(\mathbf{x}_{14}, \mathbf{x}_{18}) = \sqrt[4]{|26 - 19|^4 + |2 - 0|^4} \approx 7.01.$$
  Thus, the statement is incorrect.

## Question 3

- The explained variance is
  $$\frac{13.9^2 + 12.47^2 + 11.48^2 + 10.03^2}{13.9^2 + 12.47^2 + 11.48^2 + 10.03^2 + 9.45^2} \approx 0.87.$$
  The statement is therefore correct.
- The explained variance is
  $$\frac{11.48^2 + 10.03^2 + 9.45^2}{13.9^2 + 12.47^2 + 11.48^2 + 10.03^2 + 9.45^2} \approx 0.48.$$
  Thus, the statement is incorrect.
- The explained variance is
  $$\frac{13.9^2 + 12.47^2}{13.9^2 + 12.47^2 + 11.48^2 + 10.03^2 + 9.45^2} \approx 0.52.$$
  Thus, the statement is incorrect.
- The explained variance is
  $$\frac{13.9^2 + 12.47^2 + 11.48^2 + 10.03^2 + 9.45^2}{13.9^2 + 12.47^2 + 11.48^2 + 10.03^2 + 9.45^2} \approx 0.72.$$
  Thus, the statement is incorrect.

## Question 4

A. Such an observation will typically have a negative value as the high values
of the third and fourth coordinates will make the negative coordinates of the
principal component dominate. The statement is false.

B. For much the same reasons as in A, such an observation will typically have a
negative value. This statement is false.

C. For such an observation, the positive value of the second coordinate of the
principal component will dominate the sum in the dot product. The value will
therefore typically be positive. The statement is false.

D. The only negative coordinate of the principal component is the first one. As
the observation has a low value in this position, and high values everywhere
else, the dot product will typically be positive. The statement is correct.

## Question 5

We are given the following data:
$$n = 20000, M_{11} = 2, M_{01} = 5, M_{10} = 6.$$

The Jaccard similarity of the documents is
$$\frac{M_{11}}{M_{11} + M_{01} + M_{10}} \approx 0.1538.$$

The correct answer is A.

## Question 6

The probability $p(\hat{x}_2 = 0 \, | \, y = 2)$ can be found by marginalizing
on $\hat{x}_7$. This results in the probability
$$p(\hat{x}_2 = 0 \, | \, y = 2) = p(\hat{x}_2 = 0, \hat{x}_7 = 0 \, | \, y = 2)
                                 + p(\hat{x}_2 = 0, \hat{x}_7 = 1 \, | \, y = 2) = 0.81 + 0.03 = 0.84.$$

The correct answer is B.
