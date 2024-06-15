# Geomagnetic Storm Prediction Model

Geomagnetic storms, also known as solar storms, are disturbances of the Earth's magnetic field. They are mainly caused by solar wind waves and coronal mass ejections. In recent years, several storms have been recorded that have caused severe damage to electrical and electronic systems, satellites, communication devices, and other devices that use the Earth's magnetic field as a reference. In an era where all these technologies are of great importance in many aspects of daily life, from individuals to large companies, precise control of these extreme phenomena is crucial.

The severity of these solar storms is measured by the **Disturbance Storm-time (Dst) index**. Although significant advances have been made in recent decades in predicting the Dst through physical studies and machine learning models, there are still gaps in understanding how this data behaves.

Therefore, it is essential to develop more sophisticated models that not only predict the Dst index but also estimate its conditional density. This approach would allow a more detailed and accurate understanding of geomagnetic fluctuations, thereby improving the response capacity to potential damage caused by solar storms.

## Objectives

The main objective is to create a model for the Dst using various geomagnetic data to understand these events and prevent technological damage that could result from the unforeseen occurrence of one of these storms. More specifically, it is proposed to:

- Design and implement a model capable of explaining the Dst variable at the current time using known variables. Specifically, approximate the distribution and density of the Dst conditioned on the other observations. The model must be robust for any value, especially extreme ones.
- Accurately describe the tail of this distribution to make predictions about a threshold of the Dst value and estimate the probability of an extreme event occurring.
- Propose a family of distributions that coherently fits the real law of the Dst conditioned on the other variables.

## Data

NASA's Advanced Composition Explorer (ACE) and NOAA's Deep Space Climate Observatory (DSCOVR) satellites provide various solar wind data every minute. This data has been directly extracted from the official NASA and NOAA websites over three different time periods. The following details each of these:

- The components $(x,y,z)$ of the interplanetary magnetic field (IMF) in geocentric solar ecliptic (GSE) coordinates, measured in nanoteslas.
- The components $(x,y,z)$ of the IMF in geocentric solar magnetospheric (GSM) coordinates, measured in nanoteslas.
- The magnitude of the IMF vector, in nanoteslas.
- The latitude of the IMF in GSE and GSM coordinates, defined as the angle between the magnetic vector and the ecliptic plane, measured in degrees.
- The longitude of the IMF in GSE and GSM coordinates, defined as the angle between the projection of the magnetic vector on the ecliptic and the Earth-Sun direction, measured in degrees.
- The proton density of the solar wind, in $N/cm^3$.
- The mass speed of the solar wind, in $km/s$.
- The ion temperature of the solar wind, in Kelvin.

Due to the high correlation between solar cycles and geomagnetic storms, sunspot number data during the same time period has also been included; in this case, the records are monthly. Additionally, since the ACE and DSCOVR satellites are not stationary but orbit around the Earth-Sun Lagrange point $L1$ in a constant relative position to the Earth, the daily position of these two satellites has also been considered. Finally, the hourly Dst values calculated by the four ground observatories have been taken as the response variable of the model.

## Design and Implementation

This section describes the implemented model to approximate the conditional distribution of the Dst given the mentioned features. As shown in Figure 1, it is based on three stages: preprocessing, the neural network, and the distribution calculation. The process used in each of these is detailed below.

<div align="center">
  <figure>
    <img src="Img/architecture_model2.png" alt="Architecture of the implemented model">
    <figcaption><strong>Figure 1:</strong> Architecture of the implemented model</figcaption>
  </figure>
</div>

### Preprocessing

The two main problems that arise when working with real data like the ones we are dealing with are: diversity in the format of the different data sources and missing values.

First, the data is divided into two subsets: training and test. 20% is separated as the test set to evaluate the model's performance at the end of the process. The remaining data is used to train the model. From this point on, we work with the training set to avoid problems like "data leakage," which can lead to an incorrect and overly optimistic evaluation of the model.

We start by processing the solar wind data set. First, missing data is imputed using interpolation. This method has been chosen due to the nature of the data: being a time series, it is necessary to preserve the continuity of the data and maintain their patterns and trends. Next, to transform the data into hourly, the median of each hour is calculated. As seen in Figure 2, the appearance of heavy tails for large values in the variables presented as examples is noted. The data is not symmetrical and tends to present outliers, concluding that the mean is not a representative measure of the data. Instead, the median helps reduce the effect of these extreme values that may not be representative and allows us to handle the asymmetry of the data. Additionally, an extra variable with the standard deviation of each hour is added to understand the variability of each feature.

<div align="center">
  <figure style="display: inline-block; text-align: center; margin: 10px;">
    <img src="Img/bt_hist.png" alt="Distribution of the magnitude of the IMF" width="200">
  </figure>
  <figure style="display: inline-block; text-align: center; margin: 10px;">
    <img src="Img/density_hist.png" alt="Distribution of the proton density" width="200">
  </figure>
  <figure style="display: inline-block; text-align: center; margin: 10px;">
    <img src="Img/temperature_hist.png" alt="Distribution of the proton temperature" width="200">
  </figure>
</div>

<div align="center">
  <figcaption style="margin-top: 10px; margin-bottom: 20px;"><strong>Figure 2:</strong> Distribution of the magnitude of the IMF, proton density and ion temperature</figcaption>
</div>

Second, we merge the sunspot data. Since we only have monthly information, missing data is imputed with the most recent value. This technique is used because the primary objective of including this data is to capture the influence of the eleven-year solar cycles; the scale of a cycle is too large to notice effects on an hourly basis.

Third, we include the daily coordinates of the two satellites. Since there are observations from both ACE and DSCOVR, depending on the period, the corresponding coordinates must be merged. In cases where we do not have a reference from which satellite the information comes from, it is imputed with the most recent value.

Finally, we prepare the Dst data set by creating a second variable corresponding to the Dst value one hour earlier. This will help us predict the Dst using previous values. It should be noted the distribution of this variable, observed in Figure 3. We note the presence of non-symmetrical data with a tendency towards extreme negative values, creating a heavy tail. This is reflected in the QQ-plot in Figure 3, allowing us to conclude that we are dealing with non-normal data.

<div align="center">
  <figure style="display: inline-block; text-align: center; margin: 10px;">
    <img src="Img/dst_hist.png" alt="Distribution of the response variable, the Dst index" width="400">
  </figure>
  <figure style="display: inline-block; text-align: center; margin: 10px;">
    <img src="Img/qq_plot_dst.png" alt="QQ-plot of the response variable, the Dst index" width="370">
  </figure>
</div>

<div align="center">
  <figcaption style="margin-top: 10px; margin-bottom: 20px;"><strong>Figure 3:</strong> Distribution of the response variable, the Dst index y QQ-plot of the response variable, the Dst index</figcaption>
</div>

Finally, we normalize the data so that our model interprets each variable's contribution fairly and accurately, without being biased by scale differences.

### Quantile Recurrent Neural Network

Since we are dealing with time series, the chosen model is a Recurrent Neural Network. This way, the model can consider past information when predicting the current Dst. As mentioned, the data must be transformed into predictor sequences and have an associated response variable corresponding to the current time of the sequence.

Additionally, to approximate the distribution of the Dst conditioned on the other features, the quantile loss function is chosen (with a restriction parameter $\lambda = 10^{-4}$).

Finally, following the notation, the defined hyperparameters are: sequences of length $T = 34$, a number of neurons $K=256$, and $m=100$ output values, corresponding to each of the quantiles. In this case, a uniform partition of the interval $(0,1)$ has been considered.

It should be noted that, to ensure better generalization of the model, the use of regularization techniques could be considered. Additionally, optimizing additional hyperparameters such as the learning rate and the type of optimizer (e.g., Adam, RMSprop) could lead to improvements in model performance. However, both for computational reasons and because it deviates from the ultimate goal of this work, they have been fixed in advance.

The model is trained with 80% of the data over 500 epochs.

### Distribution Computation

Once the model has been trained, predicting a new observation results in a set of $m$ $p_i$-quantiles, for $i=1,\dots,m$. That is,

$$\hat{F}(X) = (\hat{Q}(p_1),\dots, \hat{Q}(p_m))$$

On the one hand, the points $(\hat{Q}(p_i), p_i)$ plot the distribution function. On the other hand, if we process the results following the method mentioned, the density function is plotted.

From this point, we have the necessary tools to conduct a detailed analysis and answer the questions posed at the beginning.

## Results

Next, we delve into the analysis of the distribution and density functions of a new Dst observation. First, it is shown how the robustness of the model has been verified by applying a transformation to uniform variables. Secondly, the shape of the density is studied with the goal of fitting a function $f$ to model the tail of the distribution. Subsequently, the results and their application to extreme event predictions are discussed. Finally, a family of distributions is proposed to model the global conditional distribution $Y|X$.

The approximation of the distribution and density functions of a new observation, where the Dst value at the current time is $-153$, is shown in Figure 4.

<div align="center">
  <figure>
    <img src="Img/single_dist-153.png" alt="Distribution function and density function approximated by the model" width="400">
  </figure>
  <figure>
    <img src="Img/dist_aprox.png" alt="Distribution function and density function approximated by the model" width="400">
  </figure>
</div>

<div align="center">
  <figcaption style="margin-top: 10px; margin-bottom: 20px;"><strong>Figure 4:</strong> Distribution function and density function approximated by the model</figcaption>
</div>

To verify the robustness of the model, we use the fact that any random variable applied to its distribution function results in a uniform random variable. Considering that each real value $y_i$ is a realization of the true conditional distribution and considering the previous result, the set $\{p_i \,|\, y_i = \hat{Q}(p_i), i=1,\dots,m\}$ should be a uniform sample. In Figure 5, this behavior is observed for both the training and test sets. This allows us to give some credibility to the results obtained below.

<div align="center">
  <figure style="display: inline-block; text-align: center; margin: 10px;">
    <img src="Img/uniform_train.png" alt="Histogram of the sample of probabilities for each real Dst value according to the approximate distribution, for some training set observations" width="400">
  </figure>
  <figure style="display: inline-block; text-align: center; margin: 10px;">
    <img src="Img/uniform_test.png" alt="Histogram of the sample of probabilities for each real Dst value according to the approximate distribution, for all test set observations" width="400">
  </figure>
</div>

<div align="center">
  <figcaption style="margin-top: 10px; margin-bottom: 20px;"><strong>Figure 5:</strong> Histogram of the sample of probabilities for each real Dst value according to the approximate distribution, for some training set observations y Histogram of the sample of probabilities for each real Dst value according to the approximate distribution, for all test set observations</figcaption>
</div>

Returning to the previous example, the first notable characteristic is the unimodality and improvement in the symmetry of the conditional distribution $Y|X$ compared to that of $Y$. In other words, the model provides a simplification of the Dst law, making it easier to study. However, the appearance of heavy tails suggests that the distribution is not normal (Figure 6), as suggested by the QQ-plot in Figure 6.

<div align="center">
  <figure style="display: inline-block; text-align: center; margin: 10px;">
    <img src="Img/dist_no_normal.png" alt="Comparison between the approximate distribution and a normal distribution" width="400">
  </figure>
  <figure style="display: inline-block; text-align: center; margin: 10px;">
    <img src="Img/qqplot_empir_-153.png" alt="QQ-plot of the empirical distribution relative to a normal distribution" width="400">
  </figure>
</div>

<div align="center">
  <figcaption style="margin-top: 10px; margin-bottom: 20px;"><strong>Figure 6:</strong> Comparison between the approximate distribution and a normal distribution y QQ-plot of the empirical distribution relative to a normal distribution</figcaption>
</div>

The previously mentioned fact, along with the good approximation of the left tail, allows us to fit a function $f$ that coherently approximates this tail. Given the exponential shape of this tail, it is reasonable to think that the function has the following structure:

$$f(y) = e^{a+by} +\varepsilon$$

One way to fit this function $f$ is through a logarithmic transformation and simple linear regression. Using least squares and following the explanation, the approximation is obtained, shown in Figure 7. It should be noted that this fit depends on the features $X=x$ being conditioned.

<div align="center">
  <figure style="display: inline-block; text-align: center; margin: 10px;">
    <img src="Img/points_logy.png" alt="Logarithmic relationship of the tail points, where it can be seen that the described form of $f$ is consistent" width="400">
  </figure>
  <figure style="display: inline-block; text-align: center; margin: 10px;">
    <img src="Img/tail_func_ajust.png" alt="The function \(\hat{f}\) fitted to the tail of the distribution approximated by the model, formed by the lower 15% of the data" width="400">
  </figure>
</div>

<div align="center">
  <figcaption style="margin-top: 10px; margin-bottom: 20px;"><strong>Figure 7:</strong> Logarithmic relationship of the tail points, where it can be seen that the described form of \(f\) is consistent y The function \(\hat{f}\) fitted to the tail of the distribution approximated by the model, formed by the lower 15% of the data</figcaption>
</div>

Once the conditional tail distribution function $\hat{f}$ is fitted, an extreme event can be accurately predicted:

- Determine a lower bound given a confidence level $p$.
- Determine a probability given a lower bound.

Each of these problems is formulated similarly:

$$\mathbb{P}(Y\leq y\,|\,X=x) = 1-p \iff \int_{-\infty}^y \hat{f}(s) ds = 1-p$$

In the mentioned example, given a confidence level of $90\%$, a lower bound $y=-113.49$ is obtained. Conversely, if a value below $-200$ is considered alarming enough to trigger a significant solar storm, the probability of occurrence is $p = 0.0031$.

Finally, the detailed study of a specific observation can be extrapolated to the rest of the test set observations. This is due to the similarity in the shape of the distributions of the different observations and the appearance of heavy tails in each of them (Figures 12 and 13). Therefore, it is reasonable to think that the distribution of Dst conditioned on each of the features can be modeled by the same family of distributions. Given the resemblance of the shape of each of the approximate distribution functions to the sigmoid function, it is reasonable to think that the general law of $Y|X$ can be modeled by:

$$F(y) = \frac{1}{1+e^{a+by}} + \varepsilon$$

where $F$ denotes the distribution function. Similarly to what was done to model the tail of the density, a linear model can be fitted with all values at once. Due to the good fit obtained, both for the different approximate distribution functions (Figure 13) and for the density function of a specific example (Figure 8), it is concluded that the conditional distribution can be modeled by the logistic distribution.

<div align="center">
  <figure style="display: inline-block; text-align: center; margin: 10px;">
    <img src="Img/all_quantile.png" alt="QQ-plot performed on different sequences showing indications of heavy tails" width="200">
  </figure>
  <figure style="display: inline-block; text-align: center; margin: 10px;">
    <img src="Img/all_dist_funct.png" alt="Distribution function of the logistic model fitted to the data" width="200">
  </figure>
  <figure style="display: inline-block; text-align: center; margin: 10px;">
    <img src="Img/all_dens_funct.png" alt="Density function of the logistic model fitted to the data" width="200">
  </figure>
</div>

<div align="center">
  <figcaption style="margin-top: 10px; margin-bottom: 200px;"><strong>Figure 8:</strong> QQ-plot performed on different sequences showing indications of heavy tails, Distribution function of the logistic model fitted to the data y Density function of the logistic model fitted to the data</figcaption>
</div>

---

**Theorem**: The distribution of Dst conditioned on the mentioned variables follows a logistic distribution. That is, $Y|X=x \sim \text{Logistic}(\mu(x), s(x))$ and has a density function

$$f_{Y|X=x}(y) = \frac{e^{-(y-\mu(x))/s(x)}}{s(x)(1+e^{-(y-\mu(x))/s(x)})^2}$$

where $\mu(x) = \mathbb{E}[Y|X=x]$ and $s(x)^2 = 3/\pi^2 \, \text{Var}(Y|X=x)$, and therefore depends on the sequence $X=x$ being conditioned.

---
