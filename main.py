import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

price = np.arange(9, 501, 1)
age = np.arange(0, 81, 1)
size = np.arange(3, 41, 1)

probability = np.arange(0, 101, 1)

price_l = fuzz.trimf(price, [10, 45, 110])
price_m = fuzz.trimf(price, [80, 145, 220])
price_h = fuzz.trimf(price, [190, 300, 400])
price_vh = fuzz.trapmf(price, [370, 420, 500, 500])

age_new = fuzz.trapmf(age, [0, 0, 24, 30])
age_med = fuzz.trapmf(age, [25, 30, 50, 60])
age_old = fuzz.trapmf(age, [50, 60, 80, 80])

size_s = fuzz.trapmf(size, [3, 3, 12, 16])
size_m = fuzz.trapmf(size, [13, 19, 25, 30])
size_b = fuzz.trapmf(size, [27, 32, 40, 40])


probability_l = fuzz.trapmf(probability, [0, 0, 20, 40])
probability_m = fuzz.trimf(probability, [30, 50, 80])
probability_h = fuzz.trapmf(probability, [70, 85, 100, 100])

fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(8, 12))

ax0.plot(age, age_new, 'gold', linewidth=2, label='Not old')
ax0.plot(age, age_med, 'goldenrod', linewidth=2, label='Medium age ')
ax0.plot(age, age_old, 'darkgoldenrod', linewidth=2, label='Old')
ax0.set_title('House age')
ax0.legend()

ax1.plot(size, size_s, 'gold', linewidth=1.5, label='Small')
ax1.plot(size, size_m, 'goldenrod', linewidth=1.5, label='Medium')
ax1.plot(size, size_b, 'darkgoldenrod', linewidth=1.5, label='High')
ax1.set_title('Plot size')
ax1.legend()

ax2.plot(price, price_l, 'gold', linewidth=2, label='Low')
ax2.plot(price, price_m, 'goldenrod', linewidth=2, label='Medium')
ax2.plot(price, price_h, 'darkgoldenrod', linewidth=2, label='High')
ax2.plot(price, price_vh, 'orange', linewidth=2, label='Very high')
ax2.set_title('Price')
ax2.legend()

ax3.plot(probability, probability_l, 'gold', linewidth=2, label='Small')
ax3.plot(probability, probability_m, 'goldenrod', linewidth=2, label='Average')
ax3.plot(probability, probability_h, 'darkgoldenrod', linewidth=2, label='High')
ax3.set_title('Probability of selling a house')
ax3.legend()

#Inputs
curr_price = 200
curr_age = 9
curr_size = 11

price_lev_low = fuzz.interp_membership(price, price_l, curr_price)
price_lev_med = fuzz.interp_membership(price, price_m, curr_price)
price_lev_high = fuzz.interp_membership(price, price_h, curr_price)
price_lev_very_high = fuzz.interp_membership(price, price_vh, curr_price)

age_lev_new = fuzz.interp_membership(age, age_new, curr_age)
age_lev_med = fuzz.interp_membership(age, age_med, curr_age)
age_lev_old = fuzz.interp_membership(age, age_old, curr_age)

size_lev_small = fuzz.interp_membership(size, size_s, curr_size)
size_lev_medium = fuzz.interp_membership(size, size_m, curr_size)
size_lev_big = fuzz.interp_membership(size, size_b, curr_size)

not_age_old = np.fmax(age_lev_new, age_lev_med)
not_size_small = np.fmax(size_lev_medium, size_lev_big)
not_age_new = np.fmax(age_lev_med, age_lev_old)

# Principles for getting a small probability
# 1. Price is very high and house is old and size is small
# 2. Price is high and house is old and size is small
# 3. Size is small and house is not new

l_principle1 = np.fmin(price_lev_very_high, np.fmin(age_lev_old, size_lev_small))
l_principle2 = np.fmin(price_lev_high, np.fmin(age_lev_old, size_lev_small))
l_principle3 = np.fmin(size_lev_small, not_age_new)

probability_small = np.fmax(l_principle1, np.fmax(l_principle2, l_principle3))
low_selling_probability = np.fmin(probability_small, probability_l)

print("Low probability: ")
print(probability_small)

# Principles for getting a medium probability
# 1. Price is high and house is not old
# 2. House is new and size is medium and price is medium
# 3. Price is very high and house is new and size is big

m_principle1 = np.fmin(price_lev_high, not_age_old)
m_principle2 = np.fmin(age_lev_new, np.fmax(size_lev_medium, price_lev_med))
m_principle3 = np.fmin(price_lev_very_high, np.fmax(age_lev_new, size_lev_big))

probability_medium = np.fmax(m_principle1, np.fmax(m_principle2, m_principle3))
medium_selling_probability = np.fmin(probability_medium, probability_m)

print("Medium probability:")
print(probability_medium)

# Principles for getting a high probability
# 1. Price is low and house is not old
# 2. Price is low and size is big
# 3. Price is low and size is not small
# 4. House is new and size is not small

h_principle1 = np.fmax(price_lev_low, not_age_old)
h_principle2 = np.fmax(price_lev_low, size_lev_big)
h_principle3 = np.fmax(price_lev_low, not_size_small)
h_principle4 = np.fmax(age_lev_new, not_size_small)

probability_high = np.fmax(m_principle1, np.fmax(m_principle2, np.fmax(m_principle3, h_principle4)))
high_selling_probability = np.fmin(probability_high, probability_h)

print("High probability:")
print(probability_high)

prob0 = np.zeros_like(probability)

fig, ax0 = plt.subplots(figsize = (8, 3))

ax0.fill_between(probability, prob0, low_selling_probability, facecolor='gold', alpha=0.7)
ax0.plot(probability, probability_l, 'orange', linewidth=1, linestyle='-', )

ax0.fill_between(probability, prob0, medium_selling_probability, facecolor='goldenrod', alpha=0.7)
ax0.plot(probability, probability_m, 'darkorange', linewidth=1, linestyle='-')

ax0.fill_between(probability, prob0, high_selling_probability,  facecolor='darkgoldenrod', alpha=0.7)
ax0.plot(probability, probability_h, 'yellow', linewidth=1, linestyle='-')

ax0.set_title('Activity')
#plt.show()

aggregation = np.fmax(low_selling_probability,np.fmax(medium_selling_probability, high_selling_probability))

probability_centroid = fuzz.defuzz(probability, aggregation, 'centroid')
probability_mom = fuzz.defuzz(probability, aggregation, 'mom')
probability_activation = fuzz.interp_membership(probability, aggregation, probability_centroid)

fig, ax0 = plt.subplots(figsize=(8, 3))
ax0.plot(probability, probability_l, 'orangered', linewidth=1, linestyle='-', )
ax0.plot(probability, probability_m, 'maroon', linewidth=1, linestyle='-')
ax0.plot(probability, probability_h, 'indianred', linewidth=1, linestyle='-')
ax0.fill_between(probability, prob0, aggregation, facecolor='indigo', alpha=0.7)
ax0.plot([probability_centroid, probability_centroid], [0, probability_activation], 'darkred', linewidth=1.5, alpha=0.9)
ax0.plot([probability_mom, probability_mom], [0, 0.4], 'navy', linewidth=1.5, alpha=0.9)

ax0.set_title('After aggregation')

print("Defuzz: Centroid")
print(np.round(probability_centroid, 2))

print("Defuzz: MOM")
print(np.round(probability_mom, 2))

plt.show()
