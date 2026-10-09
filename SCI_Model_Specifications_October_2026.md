# AI Engineer Brief: October Model Specifications

**From:** Chinedu Achara · **To:** Anitha Kugur · **Date:** 28 September 2026 · **Updates:** "SCI predictive models – next three specifications" (28 August 2026)

## Where we are

I have reviewed the models against what I think is achievable and usable for SCI. That review covers your work on the UK import unit-value model (Model 1) and on metal prices with Metals-API (Model 2), and our own research on predicting prices with models. It also covers the accuracy we can realistically reach.

Your work so far has been the initial research phase, and it has told us something important. Predicting the next price of a metal as a single number does not beat simply assuming the price stays the same. What SCI can use is a model that says **how far a price or cost is likely to move**, and how likely each outcome is, over the period a buyer plans for. That is close to what Model 4 (volatility and movement range) was aiming at, so that approach now runs through all three models below.

From October, please focus on these three models:

- **Model A, titanium import price:** takes Model 1 (UK import unit value) forward, applied to titanium.
- **Model B, titanium supply shift by country:** takes Model 3 (UK import supply volume by supplying country) forward, applied to titanium.
- **Model C, freight cost:** new.

The Model 2 copper, aluminium and nickel work stays. We will come back to it using the same approach.

## How SCI uses these models

Each model is a surrogate: a small, fast model of one quantity that matters to a buyer, such as the price of titanium or the cost of freight. SCI's simulation engine draws thousands of possible futures from the surrogates' outputs and runs each future through its cost calculators. That lets SCI compare supplier and material options on cost, supply risk and lead time.

For that to work, each model must give **a range with probabilities**, not a single figure. The range must be honest: wide enough to contain what actually happens as often as it claims, and no wider than it needs to be.

## Approach common to all three models

**Horizons.** Buyers of titanium and specialist alloys plan around lead times of 20 weeks or more, and around annual budgets and contracts.

- Build and test each model at **3 months and 6 months** ahead.
- Also report **12 months**. There are too few independent test periods at 12 months to judge it on its own, so it is reported but not used to decide which model is best.

**Target.** Predict the change, not the level. Use the log change between the latest published month and the month *h* ahead:

> ln(value[t+h]) − ln(value[t])

**Output.** For the models that predict a change (A and C), give seven quantiles of that change: 5th, 10th, 25th, 50th, 75th, 90th and 95th percentile. The quantiles must not cross; for example, the 10th percentile must never be above the 25th.

**Only use what was known at the time.** Every input must be something that had been published by the date of the forecast.

- HMRC trade data comes out about six weeks after the month ends.
- USGS annual figures come out in the following January.
- Time-stamp every input with the date it became available.
- Where a source revises its figures, record the date you extracted the data.

**Model types.** Research the model types suited to each problem and choose the ones you think will work best. Bear in mind that there is only around 20 years of monthly data. Talk them through with me as you go.

**Choosing and tuning.** Do all model selection and tuning on the years **before** each disruption test period.

**Calibrating the range.** The ranges must be calibrated, so that a stated 90% range contains the real value about 90% of the time. Choose the calibration method yourself.

**Baseline.** Every model is compared against "no change", with a range calibrated the same way from its own past errors. A model is only useful if it beats that.

**Testing against real disruptions.**

- Each model is tested on a real disruption: the 2022 Russian titanium sanctions, or the 2020–21 COVID freight spike.
- Fix the model, the inputs and the settings before you run the test period.
- Run the test period once, retraining at each forecast month on the data known at that date.
- Do not re-tune against the test period.
- Everyone knows these events happened, so note any choice you made knowing that.

**Use fixed random seeds,** so that every result re-runs identically from the repository.

## Model A: Titanium Import Price

**What it does.** This model predicts how far the price the UK pays for imported titanium is likely to move over the next 3, 6 and 12 months. Titanium is not traded on an exchange, so there is no published market price. The best public measure is the UK's monthly customs data: the value of titanium imports divided by their weight gives an average import price in £ per kg. This is what SCI uses to show a buyer the likely range of titanium cost for a part, and how that range changes during a disruption.

**Data.**

- **Source:** HMRC overseas trade statistics (uktradeinfo API), monthly imports into the UK, EU and non-EU combined, from 2000 or the earliest available month. UN Comtrade is the fallback; your `supply-chain-poc` Comtrade pipeline can be reused for it.
- **Titanium codes:** use the codes separately, because they are very different products and mixing them makes the average price move with the product mix rather than the market.
  - **HS 810890**, bars, sheet, forgings and other semi-finished articles: the main series, closest to what aerospace buyers purchase.
  - **HS 810820**, sponge, unwrought titanium and powder: a second series, built the same way.
  - **HS 810830**, waste and scrap: exclude it.
- **Series break:** check how the series changes in January 2021, when collection of EU import data changed.

**What goes into the model.**

- Past changes in the import price over 1, 2, 3, 6 and 12 months.
- Changes in the quantity imported over 1, 3 and 12 months.
- Each main supplying country's share of imports (see model B for how main countries are defined).
- Monthly changes in nickel and aluminium prices from the World Bank commodity data.
- The USGS titanium sponge price, as an annual figure that is only available from the January after its year.

**Test.** Choose and tune on forecast months up to January 2022. The disruption test covers forecasts made each month from February 2022 to December 2023.

**What good looks like, at 3 and 6 months each.**

- The real price change falls inside the 5th-to-95th percentile range in at least 90% of test months.
- The mean interval score (α = 0.10) is lower than the no-change baseline's. This stops a model passing by giving very wide ranges.
- A month where the model gives no output counts as a miss.
- After the disruption starts and the model is retrained, its coverage over the most recent six months gets back to at least 90% within six months.

## Model B: Titanium Supply Shift by Country

**What it does.** This model predicts whether the UK's titanium supply from each main source country is about to change sharply. In 2022, Russia's share of supply collapsed and buyers had to switch. This model gives the probability that each country's share will collapse, fall, stay stable or rise over the next 3 and 6 months. SCI uses it as the supply-risk signal for titanium: it shows how exposed an option is when it depends on one source country. It works at country level and does not score individual companies.

**Data.** The same HMRC data as model A, by partner country, using the weight (net mass) of imports, with HS 810830 scrap excluded.

**Main countries.** Any country that supplied at least 5% of UK titanium imports by weight over 2018–2020. All other countries are grouped as "other".

**Classes.**

- Take every 3-month change in each country's share from 2005 to June 2021.
- Set the limits from those changes: below the 5th percentile is **collapse**, the 5th to 33rd is **fall**, the 33rd to 67th is **stable**, and above the 67th is **rise**.
- Fix these limits before testing.

**What goes into the model.**

- Each country's past quantities and shares over 1, 3, 6 and 12 months.
- The change in total UK titanium imports.
- The titanium import price change from model A.
- Nickel and aluminium price changes.

**Output.** For each country and horizon, four probabilities, one per class, adding up to 1.

**Test.** Choose and tune on forecast months up to June 2021. The disruption test covers forecasts made from July 2021 to June 2023; Russia's collapse should show up in it.

**What good looks like.**

- The Brier score is lower than two baselines: "the same class as the last change", and "each country's historical frequency of each class".
- The probabilities are reliable: across 10 bins of predicted probability, the observed frequency is within 0.10 of the prediction.

## Model C: Freight Cost

**What it does.** This model predicts how far international freight costs are likely to move over the next 3, 6 and 12 months. Freight is part of the landed cost of every imported part. In 2020–21 container rates rose several times over and pushed up the cost of imported material. SCI uses this model to put the likely range of freight cost into each sourcing option, alongside the material price.

**Data.**

- **Preferred:** Freightos FBX Global or Drewry WCI, converted to a monthly average of the weekly values. We will use one of these if the licence is in place by 16 October; I will let you know.
- **Otherwise:** US Bureau of Labor Statistics PPI Deep Sea Freight Transportation, series `PCU483111483111`. It is monthly, free and also on FRED. Use it from 2005.
- **If you use the BLS index:** it follows contract rates more than spot rates, so it moved less in 2020–21. Show its movement next to the published FBX figures for that period so the difference is clear.

**What goes into the model.**

- Past changes in the index over 1, 2, 3, 6 and 12 months.
- Its recent volatility over 6 and 12 months.
- Monthly changes in the Brent crude price from the World Bank commodity data.

**Test.** Choose and tune on forecast months from 2012 to December 2019. The disruption test covers forecasts made from January 2020 to December 2021.

**What good looks like, at 3 and 6 months each.**

- The real freight cost change falls inside the 5th-to-95th percentile range in at least 90% of test months.
- The mean interval score (α = 0.10) is lower than the no-change baseline's. This stops a model passing by giving very wide ranges.
- A month where the model gives no output counts as a miss.
- After the disruption starts and the model is retrained, its coverage over the most recent six months gets back to at least 90% within six months.

## Working Together

Keep the work in GitHub as you go. We will review the models together as they develop, and you can talk me through the results as each one reaches its disruption test.
