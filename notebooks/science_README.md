# Science Experiments

Before jumping in, are you at the right place? If you wish to generate a defensible and statistically rigorous analysis
of the characteristics of an intervention against a control, then you are at the right place. However, if you wish to
simply run a grid search over some number of parameters for exploratory purposes, then you should instead use the
`grid_search` tool under the sweeps folder.

## Background

Science requires a mindset shift from engineering. Our goal is no longer to make what isn't; it's to characterize what
is, which requires a shift in priorities. First, don't be wrong. Second, be right without spending a lot of money. Both
are impossible, naturally, but with great care we can make tiny bits of progress which would be better than I'm doing.
Recall that the goal cannot be to show that a candidate is universally better than another. Science can't prove a
general theory to be true. In practice, it's very difficult to show that a candidate theory is even better than another
and that's where these tools come in. An easier and more appropriate place to start is to characterize the performance
of a candidate across some number of dimensions such as peak learning at a specific number of agent steps or wall-clock
time, speed of initial learning, variance, hyperparameter sensitivity, etc. compared to a baseline and then point to
specific situations when one is a better practical choice than another. Scientific testing posits that claims as to
_why_ a candidate performs differently generalize better than, say, ranking candidates. To reiterate, each recipe tests
for something _specific_. ABES tests for rapid learning, which might show that a bigger, more-expressive model performs
worse! In fact, a general heuristic is that more expressivity and leaving more parameters up for the bitter lesson to
decide leads to worse performance in early testing, only to invert with enough data. This informs what you can say when
looking at results in ABES at, say, 2b time steps vs 50b or with and without kickstarting. Similarly, Navigation favors
certain biases that ICL may or may not. This is a counter to the instinctive desire to show that either A) our new
candidate is generally better or B) the existing SOTA has a failure mode that our new candidate does not. A is discussed
above and, for B, it is exceptionally challenging to show that an algorithm has a consistent failure case. We can always
ask: would this happen for a different choice of hyperparameters?

Almost always, the answer becomes "gather more data points" which is, of course, expensive and time-consuming. In
determining the needed level of rigor, consider the cost of being wrong. The cost is roughly the magnitude of the
negative outcome along with the likelihood that an invalid result would be detected. If both are high then we need to be
very sure about our claims. This isn't meant to say that we always need this. In practice, we very often "manually"
sweep and take far fewer samples than we need for statistical certainty. This is often fine since you're just hurting
yourself (and not me (and we wouldn't be humans if we didn't hurt ourselves)). It's a practical approach to gaining
rough and sometimes incorrect insight about a search space that is simply too big.

As important as it is, the most critical thing isn't the statistical technique; it's generating good data to which we
can apply said techniques. So we put great effort into getting the setup right. Then we spend the money on the runs,
doing it with as few biases as possible, and hopefully outputting sound data that we can then analyze in a number of
ways depending on the profile we are trying to understand.

## Guidelines for Generating Useful Data

Answering the question of whether the control is better or worse than the hypothesis requires **agreeing to the task**.
Here, we agree that the task reflects what the company is trying to solve which, in practice, means solving the recipes
the Cybernetics team is using as well as some basic variations of arena as of October 2025. This will eventually shift
to CvC tasks.

#### Tasks/Recipes

**October 2025 Recipes**

1. Arena basic easy shaped - `arena_basic_easy_shaped.py` - 5b agent steps
2. Arena with sparse rewards - `abes_prog_7.py` - 5b agent steps
3. Navigation - 10b agent steps (! remains to be verified !)
4. ICL - 5b agent steps (! remains to be verified !)

As this list evolves in the future, we need to keep an eye open for tests that are highly correlated (hence, redundant)
and remove them as it is a waste of compute.

**Versioning** becomes a major issue - we don't want to have to regenerate controls every time. Worse, we want to be
able to compare results from one run to results from another run. As such, these tasks are under a folder with the date
as the version identifier.

We chose to go forward with **curriculum** despite how intensely path-dependent it is because, it is considered a
fundamental part of the task.

The **action space shouldn't have attack**. It reduces reward, making it difficult to tell which policy is better. We
could use NPCs eventually to test which policies are better at attack during an eval but agents aren't currently trained
with NPCs in their envs. Finally, cogames doesn't have attack anyway. For now, train without attack and run eval (if
relevant - see below) without it as well.

**Architectures** should vary. We propose two:

- An LSTM with its prior over memory
- A transformer type with investigations being things like can a larger or more expressive model perform better later in
  training even if starting up more slowly, can an intervention make it startup faster, etc.

#### Selecting a **summary statistic** is a design choice.

If investigating early speed of learning then it's recommended to use area under the curve (**AUC**). Ensure that the
AUC cutoff isn't too late in training; ideally it ends right when agents are at the end of their grokking phase but
before they start diffusion. Note: High learning rates can improve this metric but cause poor generalization and
instability later in training.

However, peak learning is generally a preferred characteristic to investigate. If investigating peak learning then use a
**series of evals**.

- They would ideally be over a test suite of held-out env params but still within the sampling distribution.
- This requires that training must be over a domain of configuration parameters. Domain randomization should also be
  standard practice anyway.
- When cogames is up, evals will become more fully specified.
- There must be enough evals; an eval is run on a single checkpointed policy but that checkpoint captures a single point
  in the training run, representing a single state along the many states the policy takes as it's updated in training.
  As such, we need to take many checkpoints and eval them and finally summarize those evals across many checkpoints into
  a single number to represent the outcome of the seed. In practice, this looks like taking many checkpoints towards the
  end of training (much more frequently than you would otherwise) and running evals on those checkpoints. Then taking
  the mean of those evals to be the summary statistic. If evals can't hit the goals above and you wish to characterize
  peak learning then you might as well use the AUC over a period that is limited to late training.

**Discarding or replacing runs:** sometimes the policies statistics will diverge for one or more runs, oftentimes
crashing down to near zero or oscillating wildly. Do not just remove this run or run it again with a different random
seed! Why did it diverge? Were the hyperparameters too aggressive, resulting in poor performance? Investigate and report
this. Outlier analysis is a method of investigating these phenomena.

#### Variation Over Time and Seeding

The **number of agent steps** depends on the map/recipe and the research goal. Some recipes converge faster than others.
If you are targeting the early speed of learning then obviously you need fewer steps. However, we don't know how far out
we need to train to understand peak learning. The recommendations here are based on what we've seen on previous recipes.
New interventions could someday afford learning well past these recs so keep an eye open for an upward slope at the
cutoff (how exciting would that be!). Consider tracking the average derivative over the last 1 million agent steps.

The **necessary number of seeds** depends on the seed-to-seed variability and that is a function of the env. Hence, it
differs based on the recipe. For instance, `arena_basic_easy_shaped` has a pretty tight grouping whereas `abes_prog_7`
has a massive spread. We ran a test of a number of seeds on each recipe to determine the spread and gave our
recommendations above.

**Warning:** be extra wary of tests that have binary outcomes or where most agents fall near one end or another. If
given some small number of samples, there is a non-negligible chance that all samples are on one side of the outcome
and, as a result, our statistical tools will never see a single example from the other outcome. This will lead to
invalid confidence intervals!

Whenever possible in experimentation try to use _repeated measures_. Here, that means: try to use **paired seeds** for
the initialization of the weights of the network. Generate a single seed and use it both as the control _and_ your
candidate intervention. Obviously, this may not be feasible if you are testing different architectures but it is a good
way to increase statistical power or reduce the number of needed runs if testing a new loss function or something
similar. If you wish to use the same seeds as those used in our control runs then set that parameter in the experiment
recipe. Future work will investigate pairing the same environment seed as well. Keep in mind that setting the system
seed to the same number can fail to do what you think it does if, in one path, a different set of instructions are run
by the time the seed generator is used to generate your weights. Adding additional calls to the generator ahead of where
you wish it to generate the same random numbers will not get you what you want. The best way to solve this is to write
unit tests against where you expect to have the same random numbers.

Note that even with the same weight initialization (seed), there is notable variation. We made three training runs on
`arena_basic_easy_shaped` using the exact same weight initialization (seed) and found variation at 2b time steps to be
at least 10% due to env path dependence and other factors. We have yet to run this test using the same env seed.

If you select for the best seeds and run your experiment on that subset then you will have your license revoked.

#### Hyper Hypotheses

This is one of the most important parts and, yes, you have to sweep. As the Bloominism goes, "RL is notoriously
sensitive to hyperparameters." We put a lot of effort into lowering the dimensionality of the other terms to make room
for this but it's also the harsh reality - the hypers that we are so confident about and the ones that you think are not
a part of your candidate intervention may indeed need to be updated due to coupling with the new hypers your have
introduced. Whether you simply added an extra output head or replaced a layer, the learning system remains highly
coupled and nonlinear. In fact, you will also have to re-do the control (sweep over it again) if you change something
about the environment or curriculum and this is the kind of thinking needed for paper-quality experiments.

Fortunately, however, Axel has anecdotally found that the performance curve over various values for a given
hyperparameter seems to be only weakly coupled with the other hyperparameters under test. If your test has a low cost of
being wrong then consider only sweeping over your new hypers, leaving the others fixed. Once those hypers are fixed only
_then_ can you begin data-generation.

A note on our business decisions: Each environment demands a unique set of hypers to maximize it. However, we prefer to
have one set of hypers that work best toward our ultimate, singular goal: cooperation. We recommend tuning a single set
of hypers across the recipes articulated above with the assumption that they, together, represent different if limited
views of the goal of cooperation. This may become even more focused as cogames matures.

If the best hyper is at the edge of the range swept over then you need to expand the sweep in that direction.

Keep **maximization bias** in mind. Concisely put, the candidate with more tuned hypers has an unfair advantage in a
generic maximization sweep. What this means in practice is that:

1. If you ran a sweep, selected hypers and _then_ began running your experiments/generating data with these fixed hyper
   values then you are all good.
2. If you use the output reward values of the sweep as part of your data then you need to account for maximization bias
   or increase the effect size you are looking for. It's a much more complicated procedure and is discussed in the
   appendix.

#### Application

Run the experiment tool in each experiment recipe, found in the experiments folder (the specific experiment folder will
have a date as a version number). You simply need to set the number of runs, whether to use the same seeds and then add
your intervention parameters to the experiment recipe to therefore generate the runs for your candidate intervention.
Once training completes, run the analysis script as explained below.

## Analyzing Good Data

Having survived the pain of setting up and running a slightly less wrong experiment, we get to actually take conclusions
from our analysis!

The analysis tool takes the run names in two groups, control and candidate intervention. The control group has a default
list which points to our carefully constructed control runs if you had the option to compare against, and hence re-use,
our data. If you had to generate your own control (i.e. had to use the experiment tool once for your candidate
intervention runs and once to generate your custom control runs) then enter the appropriate run name, overriding that
arg.

It then uses the wandb API to download the data and run the analysis you want. You can specify which summary statistic
to use and the method of comparison and it will generate it for you. Depending on the method you chose, it will also
attempt to run tests on the assumptions inherent in the method and print the results to your console.

If you were able to use the same seeds then enter the paired run names along with their seed number. Yes, this is
redundant but it's so that it's harder to enter these params in error.

#### We Already Have Your Controls

To save compute, we carefully generated a set of control runs based on commonly used settings so you only have to
generate runs for your candidate intervention. The run names are listed in the test recipes along with their seed
numbers.

#### A Flow Chart of Methods

##### Are there differences?

Ideally, we can use a **t-test** due to its statistical power but it carries assumptions that are often not met in RL.
If this is selected, we'll automatically run tests for normality and homoskedasticity, default to unequal variance
(Welch's t-test, not Student's t-test), detect whether you want a paired t-test or not, and then print the results to
the console. Your data may not conform to these assumptions and a result will not print if normality is not met.
Further, reporting p values is fraught and it is preferred that you report intervals, not p values.

A more robust option is to use a **bootstrap confidence interval** using **resampling** but it has weaker statistical
power and hence requires more runs. It makes no assumptions about normality or homoskedasticity and outputs a range for
which the data falls within a confidence interval (see below). **This should be your default option.** If the bottom of
the range is above 0 then your intervention has a positive effect within the confidence interval.

We can use this for both the paired seed case and otherwise. If paired, we compute the difference in summary statistic
values for each pair, creating a set. We then sample with replacement 10,000 times (we have not investigated why 10,000x
is the number used in literature) and take the confidence interval out of that distribution.

If seeds are not paired, we take each of the two sets of summary statistics (one for the control and one for the
intervention), resample with replacement, compute the mean of each set, then report the difference. We repeat 10,000
times to present an interval.

Finally, we output a **Bias-Corrected and accelerated** (BCa) interval. Note that resampling assumes that your samples
span the distribution. Testing for whether this is true is difficult and the literature we've found simply recommends
15-20 samples. See the appendix for alternative methods.

##### Do we have the statistical power to make these claims?

We run a basic analysis to determine how many runs we need for a desired effect size and significance level. We run this
test _post hoc_ which is frankly not appropriate. We can benefit from more sophisticated power analyses and to also
check bootstrap intervals for smoothness and variance in the future.

#### Additional Parameters

The significance level, `alpha`, is the probability of rejecting the null hypothesis when it’s true (Type I Error). It's
set to the usual 0.05 but it might make sense to set it to 0.10 if implementing a candidate that has external
verification. Further, we can justify a one-sided confidence interval since we typically only look for improvements and
don't care to really prove that a candidate intervention is worse. This increases power or reduces the number of runs
needed.

Power level, `beta`, is the probability of correctly rejecting the null hypothesis when it’s false. It's set to the
usual 0.8.

The effect size, `target_effect_size`, we are looking for defaults to 0.2 which is not small. Going higher helps reduce
the number of trials needed and justifying changes to the codebase is a non-trivial cost.

Future work can include a tolerance interval if we wish to find the expected performance range for x% of all runs.

## TL;DR: how me run experumints?

Now that you have taken great care going through the above,

1. Run sweeps over your candidate with the appropriate level of rigor
2. Run the experiments tool (a modified grid_search tool) once per recipe with the parameters filled out as per your
   careful analysis. Again, do this if you need to build your own control.
   1. Params: recipe, seed, number of runs, agent step count at termination
   2. Fixed params: (as discussed above) things like no attack
3. Wait.
4. Once the experiments have completed, analyze them by running the analysis script by running
   `uv run ./tools/run.py experiments.analysis.compare` and fill out the necessary parameters. See the example code
   block below. The output will print to the console. Additional reporting options are conveniently under the "Future
   Work" section which doesn't exist. If you don't like the analysis options, the code is written such that you can
   readily add your own methods, keeping complaints to a minimum. Params include things like summary statistic, run
   names in groups or paired run names (one or the other), analysis method(s) and additional parameters. Example usage:

`uv run ./tools/run.py experiments.analysis.compare \ ` `summary.type=auc \ `
`summary.percent=null summary.step_min=8000000000 summary.step_max=10000000000 \ `
`fetch.samples=None fetch.min_step=8000000000 fetch.max_step=10000000000 \ ` `fetch.keys=["overview/reward"] \ `
`pairs='[{"control":"ppo.seed42","candidate":"not_ppo.seed42"}, {"control":"ppo.seed123","candidate":"not_ppo.seed123"},]`

## Appendix

#### A1. Maximization Bias

**Maximization Bias** If we allow one algorithm to have more hyperparameter settings, then differences in performance
can be due to maximizing over more hyperparameter settings rather than differences in the algorithm. This is only true
if we include the performance results (ie reward) of our hyper parameter sweep in our reporting of our candidate scores.
The bias is the result of reporting the _max_ performance of a sweep over hyper values.

One solution is a two-stage approach whereby we 1) run a sweep over hyper values and then fix the values. Then 2) run
many more runs over the fixed hypers. We then only report perf from the second stage. This is a great way to avoid being
wrong. However, it's wasteful of compute since we throw away the first stage and it may not converge on the best hypers
because we have to spread our compute across sweeps as well as data generation for our experimental comparisons.
Nevertheless, this is the approach we recommend above because its simpler to implement.

If compute were no issue, we would break up the repeated seeds on the first stage and compute statistics for each group.
This is a search over H hypers with N runs each. We then repeat the process M times. We then review if the reported
reward changes across the group and gain an understanding of how sensitive hypers are to reward. This is an HxNxM space
where N is the number of seeds and M is the outer loop. A better tactic is to use **nonparametric bootstrapping**. Here,
M=1 (we simply run a batch N times) and then bootstrap N many times to calculate many means and variances. From there,
we look at the many means to understand the variance in the search process. The authors of _Empirical Design in
Reinforcement Learning_ recommend N = 10 instead of the usual N = 30.

Making this change involves some integration into sweeps. It also needs to be considered in the generation of the
control: we must ensure that we test each candidate over the same number of seeds, even if one has fewer hyperparameter
options than another. Without knowing how many hyperparameter options future candidate interventions will have, we would
sweep the control over more seeds that we would otherwise, creating room for future candidates. We then note how many
seeds we used for the control and make that the available run quantity for the candidate's seeds. This is in the
category of paper-quality experimentation, not necessary for exploratory work.

Of course, maximization bias is not limited to hyperparameters but pertains to anytime you select the max of any metric.

#### A2. Alternative Methods

Optionally, you can augment tests with a rank-based test such as a Brunner-Munzel test. Only do this if you need the
scale on which this rank-based test outputs: a bootstrap confidence interval gives a range in the same unit as your
summary statistic while a rank-based test like BM gives a probability of stochastic dominance. If the two tests give
differing results then you may need a Bonferroni correction. Be wary of this version of "p-hacking."

Future work will include a mixed-effect model.
