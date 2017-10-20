mope_map.params contains the parameter file for the maximum a posteriori
parameter values from the application of mope to the data from
Rebolledo-Jaramillo et al. (2014). Note that parameter files are required for
simulations only.

The file is separated into two sections, before and after a line starting with
'---'. Before this line, the parameters of genetic drift and mutation are
given. The first value on these lines is the name of the parameter pair, the
second value is the genetic drift parameter, and the third parameter is the
mutation rate parameter.

Parameters for ages are given after this separating line. For these parameters,
the first value is the mean, and the second value is the standard deviation. In
simulations, the ages are assumed to be normally distributed with these
parameters. Note that these parameter names must correspond to age values given
in the .newick ontogenetic tree file. These distributions are truncated on the
left tail at zero. The final line of ages conveys the relationship between
mother age and mother birth age, namely that the mother's age (at sampling,
here) is the age of the child plus the age of the mother at childbirth. 

```
blo  2.104e-3       1.035e-8
buc  2.08e-3        1.0635e-8
fblo 2.104e-2       1.035e-6
fbuc 2.08e-2        1.0635e-6
som  4.937e-2       1.0632e-8
loo  1.984e-4       1.0265e-8
eoo  3.903e2        1.01626e-8
emb	 3.14697e-3     2.5575e-3
----------------
mother_birth_age  30.30 5.11
child_age         10.16 5
mother_age        child_age+mother_birth_age
```
