# Data Sources, Attribution And Publication Review

Reviewed against provider guidance on 13 October 2026. This records source guidance, not legal clearance for every downloaded file or derived artifact. Do not assume that free downloading grants unrestricted public redistribution. Confirm the specific series and your intended use before a public repository release. Private sharing is not automatically exempt from provider terms.

| Source | Used for | Guidance and remaining check |
| --- | --- | --- |
| United Nations Statistics Division, UN Comtrade | A import unit values/weights and B supplier shares; prepared history starts in 2010 | [Data-use policy](https://uncomtrade.org/docs/policy-on-comtrade-data-use/) and [redistribution guidance](https://uncomtrade.org/docs/re-dissemination-of-data/). Review the applicable agreement for public redistribution of prepared panels and derived outputs; internal-use permission alone is insufficient evidence. |
| Bank of England Database | A/B GBP-USD exchange-rate conversion and features | [Bank legal terms](https://www.bankofengland.co.uk/legal). Database information is generally under the Open Government Licence, but selected exchange-rate series are excluded because of third-party licensing. Confirm whether the actual GBP-USD series in these panels is covered before publishing it or its derivatives. |
| UK Office for National Statistics | A/B ERUC import-price index features | [ONS reuse guidance](https://www.ons.gov.uk/methodology/geography/licences/) and [Open Government Licence v3](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). Check dataset-specific notices and third-party exclusions; include the required attribution. |
| US Bureau of Labor Statistics, obtained through FRED PCU483111483111 | C deep-sea freight producer-price index | [BLS copyright information](https://www.bls.gov/opub/copyright-information.htm), [FRED series](https://fred.stlouisfed.org/series/PCU483111483111). BLS-published data are public domain apart from stated exceptions; retain BLS/FRED attribution and check the series-specific notes. |
| World Bank Commodity Price Data (Pink Sheet) | C Brent oil input; workbook includes other unused commodity series | [Commodity dataset](https://datacatalog.worldbank.org/search/dataset/0038238/commodity-prices-history-and-projections), [dataset terms](https://data.worldbank.org/summary-terms-of-use), [download page](https://www.worldbank.org/en/research/commodity-markets). Check the dataset licence and third-party exceptions; attribute World Bank and identify processing changes. Do not apply one licence blindly to every sheet in the workbook. |

## Attribution Record

Prepared data derive from UN Comtrade (United Nations Statistics Division), Bank of England Database, Office for National Statistics, US Bureau of Labor Statistics via FRED, and World Bank Commodity Price Data, as applicable to each model. Changes include monthly aggregation, supplier grouping, currency/unit conversion, feature engineering and shifted supervised targets. The providers do not endorse these models or forecasts. Source downloads and derived rows may be revised over time.

Suggested notices, subject to series-specific confirmation: ONS data contain public sector information licensed under the Open Government Licence v3.0; Bank database data are copyright the Governor and Company of the Bank of England where applicable. Do not use the latter statement to override the exchange-rate exception.

## Remaining Publication Decisions

- Confirm the exact Bank of England FX series and its redistribution status.
- Confirm UN Comtrade redistribution conditions for the prepared panels/results and your repository audience.
- Retain source-specific metadata/notices for the Pink Sheet workbook and FRED series.
- Choose a code licence with the repository owner's approval; no licence for user-owned code was assigned automatically.

No data files were removed or published during this packaging work. Technical inference verification does not verify these permissions.
