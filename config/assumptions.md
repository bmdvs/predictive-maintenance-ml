# Assumption register

One row per mechanism parameter in `config/config.yaml` (plant behavior). Columns:
- **Current:** config v11, run 06.
- **New:** config v12, revision 2.
- **Published range** and **Source:** what was found. A source marked "(snippet)" was seen in search results but not opened, and still needs a direct check.
- **Verdict:** one of four values.
  - **Supported:** the value sits inside a published range.
  - **Consistent:** published sources give the direction or order of magnitude, but no tight range.
  - **Engineering assumption, no published benchmark:** nothing transferable was found.
  - **Shop record:** the value describes this shop, not equipment behavior.

The README and reports never cite a source for an engineering assumption.

Sources were searched on 2026-10-29.

## Bearings

| Parameter | Current | New | Published range | Source | Verdict |
|---|---|---|---|---|---|
| Stage 2 to failure, spindle (beta) | [0.0065, 0.0013], about 7.5 weeks | [0.0048, 0.0010], about 10 weeks | Envelope warning on outer-race defects 3 to 6 months; inner race "often shorter"; cage days to weeks | Reliability Magazine, Early Failure Detection, https://reliamag.com/articles/early-failure-detection/ | Consistent: inside the published span. The R1 seed's "vibration 1 to 9 months" was not found in any source |
| Stage 2 to failure, TSC pump | [0.0270, 0.0054], about 4 weeks | [0.0134, 0.0027], about 8 weeks | As above | As above | Consistent |
| Stage 2 to failure, flood pump | [0.0162, 0.0032], about 4 weeks | [0.0081, 0.0016], about 8 weeks | As above | As above | Consistent |
| Stage 2 to failure, compressor | [0.0061, 0.0012], about 8 weeks | [0.0041, 0.0008], about 12 weeks | As above | As above | Consistent: about 3 months, the low end of the envelope range |
| Four-stage progression (signatures) | Stage 1 HF only; 2 envelope; 3 spectrum with sidebands; 4 broadband | Unchanged | Stage 1 ultrasonic HF; 2 resonance 500 to 2,000 Hz with sidebands; 3 defect frequencies and harmonics; 4 broadband floor | Acoem, 4 stages of bearing failure, https://www.acoem.com/en/blog/do-you-know-the-4-stages-of-bearing-failure/ | Supported (qualitative) |
| Stage 3 temperature rise | None | 3 to 5 C | Qualitative: "significant temperature increase" at stage 3, temperatures "skyrocket" at stage 4. No delta-T thresholds for bearings: the Infraspection Institute says temperature cannot predict time to failure, and the ITC and NETA delta-T tiers are for electrical equipment | Machinery Lubrication, https://www.machinerylubrication.com/Articles/Print/29152 ; Infraspection, https://irinfo.org/tip-of-week-alphabeta/temperature-rise-as-a-severity-indicator | **Engineering assumption, no published benchmark** (direction supported). The R1 seed "ITC stage 3 5 to 15 C, stage 4 over 25 C" was not confirmed |
| Stage 4 temperature rise at failure | 9 C, all classes | Spindles 12 to 15 C; pumps and compressors 15 to 20 C | As above | As above | **Engineering assumption, no published benchmark** (direction supported) |
| Spindle front-set life | Older 18,500, newer 25,000 h | Unchanged | High-speed spindle bearings 10,000 to 20,000 h; motorized-spindle precision life about 8,000 h | USPTO patent 10286514, https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10286514 (snippet, unopened: 10,000 to 20,000 h for high-speed spindle bearings); Scientific.net AMM.421.81, https://www.scientific.net/AMM.421.81 (snippet, unopened: motorized-spindle precision life about 8,000 h) | **Engineering assumption, no published benchmark** (step 2 decision: kept as configured). The current values (18,500 and 25,000 h) sit at or above both unopened figures. The R1 seed's "20,000 to 30,000 h design-life table" was not found |
| Spindle rear-set life | Older 30,500, newer 39,000 h | Unchanged | None | | **Engineering assumption, no published benchmark** |
| TSC pump motor bearing life | Older 7,500, newer 10,000 h | Unchanged | None | | **Engineering assumption, no published benchmark** |
| Flood pump motor bearing life | Older 9,500, newer 13,000 h | Unchanged | None | | **Engineering assumption, no published benchmark** |
| Compressor motor bearing life | Older 22,000, newer 28,000 h | Unchanged | Regrease or replace every 4 years or 16,000 h | Ingersoll Rand UP6S procedure (community-written, cites no manual), https://www.getmaintainx.com/procedures/d/zJ5TkAo4MvA/ingersoll-up6s-11-22-kw-oil-flooded-rotary-screw-compressors-up-series-every-4-years-or-16-000 | Consistent: a maintenance interval, not a life. Weak source |
| Compressor airend characteristic life | Older 27,000, newer 38,000 h | Older 40,000, newer 50,000 h (older was 39,000 in config v12; set to the quoted industry figure at the pilot04 review, 2026-10-29) | Quincy marketing sheets, not a characteristic life: QSI 245i to 500i, "Over 130,000 hours of operation (Industry Standard = 40,000 hrs)"; QSI 90 to 225, airend "designed to last over 100,000 hours" | Quincy QSI sheets via distributor, opened 2026-10-29: https://burtoncompanies.com/wp-content/uploads/2026/02/QSI-245i-500i.pdf ; https://burtoncompanies.com/wp-content/uploads/2026/02/QSI90-225.pdf | Consistent. Older 40,000 h matches the quoted industry standard; newer 50,000 h sits between it and Quincy's own claims. The run-06 values (27,000 and 38,000 h) were both below it |
| Onset Weibull shape | 2.1 | Unchanged | None | | **Engineering assumption, no published benchmark** |
| Spindle rebuild interval, older HMCs | 20,000 h | Unchanged | None found for interval rebuilds | | **Engineering assumption, no published benchmark** |
| Inspection detection probability by stage | 0.10, 0.20, 0.70, 0.95 | 0.03, 0.20, 0.70, 0.95 | None | | **Engineering assumption, no published benchmark** |

## Crashes and spindles

| Parameter | Current | New | Published range | Source | Verdict |
|---|---|---|---|---|---|
| Crash rate per machine-year | 1.5 base, about 3.6 effective | Unchanged | None | | **Engineering assumption, no published benchmark** |
| Crash damage factors and outcome shares | 0.3 to 0.7 life factor; 0.975, 0.015, 0.010 | Unchanged | None | | **Engineering assumption, no published benchmark** |
| Crash share of HMC spindle failures (check 29) | Not checked | Reported against 30 to 60% | About 33 to 60%. Maier and Kimmelmann: "about 85 % of spindle failures result from bearing damages", of which 40% are due to machine collisions (about 34% of all), almost 50% wear, 10% dirt or insufficient lubrication; they also cite other surveys at 35% and 60% of spindle failures caused by a crash. Vendor survey 33%; trade article about 60% | Maier and Kimmelmann, J. Machine Engineering 17(1) 46 to 56, 2017, opened 2026-10-29 from the publisher, https://wydawnictwo.not.pl/2017JOM/V1/4_MAIER.pdf (the BazTech copy, https://yadda.icm.edu.pl/baztech/element/bwmeta1.element.baztech-41860fe2-cbb3-4417-abfa-2eb70236a067/c/4_Maier_Kimmelmann_spindle_journal_of_mach_eng_1_2017.pdf , still returns 403); Motorex survey, https://www.siams.ch/news-en/motorex-spindle-line/496 (snippet); Design World, https://www.designworldonline.com/?p=14399 | Supported. The paper itself gives 34% derived and cites 35% and 60%, so the seed's 35 to 60% is confirmed; with the 33% vendor survey the span is about 33 to 60%, inside the 30 to 60% check range |
| Spindle rebuild out, working days | 3 to 7 days (no spare model) | 15 working days out; swap 3 to 7 days; 10 to 21 working days without a spare | Quick turnaround (QTA): evaluation "within five business days of receipt"; spindles "returned in 7 to 10 business days after approval" (page prints a dash), about 12 to 15 business days from receipt plus approval time. The current page gives no standard-repair figure; the "standard 10 to 20 business days" appears only in search summaries of an earlier version and is not on the page as opened. Setco's rapid repair and spindle repair pages give no number either | Setco QTA page, opened 2026-10-29, https://www.setco.com/?p=72985 ; Setco spindle repair, https://www.setco.com/services/spindle-repair-rebuild/ ; Setco rapid repair, https://www.setco.com/?p=75430 | Consistent: the 15 working day rebuild sits at the top of Setco's quick-turnaround path. The 10 to 21 working days without a spare is not confirmed by a standard-repair figure on Setco's current pages and is accepted as an **engineering assumption, no published benchmark** (pilot04 review, 2026-10-29) |
| Spare spindle count | None | 1 for 8 HMCs | None | | Shop record (engineering assumption) |

## Lubrication

| Parameter | Current | New | Published range | Source | Verdict |
|---|---|---|---|---|---|
| Episodes per spindle-year | Older 2.0, newer 0.75 | Unchanged, constant over time | No spindle-specific rate. Lubrication accounts for 36% of premature bearing failures in general (snippet) | Reliability Magazine, bearing failure cause statistics, https://reliamag.com/topics/guides/bearing-failure-cause-statistics | **Engineering assumption, no published benchmark** |
| Temperature rise | 3 to 8 C | 3 to 10 C | Qualitative only | Machinery Lubrication (as above) | **Engineering assumption, no published benchmark** |
| Inspection detection | 0.75 flat | 0.75 at a rise of 5 C or more, 0.25 below | None | | **Engineering assumption, no published benchmark** |
| HF rise, ramp, recovery, conversion window | 30 to 80%; 5 to 15 d; 1 to 3 d; 20 to 40 d | Unchanged | None | | **Engineering assumption, no published benchmark** |
| Lube PM records the finding (M9) | No | Yes | | | Shop record |

## Coolant pumps and rotary unions

| Parameter | Current | New | Published range | Source | Verdict |
|---|---|---|---|---|---|
| TSC pump type and pressure | 69 bar | Unchanged; described as positive displacement, clearance wear | 70 bar (1,000 psi) at 28 L/min for through-tool coolant, Hydra-Cell diaphragm pump | Wanner application story (snippet), https://wannerpumps.com/wp-content/uploads/partner-portal/uk-documents/application-stories/high-pressure-coolant-for-through-tool.pdf | Supported |
| Seal wear onset and growth | 0.30 per pump-year; 60 to 180 days | Unchanged | Average seal life 3 years or more with good programs; API 682 target 25,000 h; 60-month site benchmark | Pumps & Systems, https://pumpsandsystems.com/how-long-will-mechanical-seal-last-part-one-two ; Reliability Magazine, https://reliamag.com/topics/guides/pump-reliability-mtbf-statistics (snippet) | Consistent: 0.30 per year is about a 3.3 year mean |
| Clearance or impeller wear onset | 0.15 per pump-year | Unchanged | None | | **Engineering assumption, no published benchmark** |
| Cavitation frequency | TSC 3.0, flood 2.0 per pump-year | Unchanged | None | | **Engineering assumption, no published benchmark** |
| Cavitation wear multipliers, TSC (M5) | None | Onset hazard x3, progression x2 while cavitating | Dry running damages seals and positive-displacement pumps (qualitative) | World Pumps, https://www.worldpumps.com/content/features/how-to-overcome-the-challenge-of-dry-running (snippet) | **Engineering assumption, no published benchmark** (direction supported) |
| Seal failure rate and wear multiplier | 0.25 TSC, 0.10 flood; x3 | Unchanged | As for seal wear | As above | Consistent |
| TSC replacement interval | 730 days | Unchanged | None | | Shop record |
| Rotary union life | 4,000 TSC running h, shape 2.0 | Unchanged | No life in hours; Pop-Off seals allow unlimited dry running | Deublin, https://www.deublin.com/-/media/API-Sync-Assets/CAT/Rotary-Union-Catalog-for-Machine-Tool.pdf?dmc=1 (snippet) | **Engineering assumption, no published benchmark** |
| Ingress fraction, hazard multiplier | 0.4; x2 to x4 | Unchanged | Contamination is a leading cause of spindle bearing damage (qualitative; Maier and Kimmelmann give 10% dirt or lubrication) | Maier and Kimmelmann (as above) | Engineering assumption (direction supported) |

## Spindle cooling

| Parameter | Current | New | Published range | Source | Verdict |
|---|---|---|---|---|---|
| Onset per HMC-year | 0.35 | Unchanged | None | | **Engineering assumption, no published benchmark** |
| Gradual residual and ramp | 2 to 6 C over 2 to 8 weeks | Unchanged, 70% of faults | Cooling-channel design changed bearing temperature by up to 6.7 C; doubling shaft-core flow halved the temperature rise | Applied Sciences 2020, https://www.mdpi.com/2076-3417/10/11/3991 ; Xi'an Jiaotong 2018, https://zkxb.xjtu.edu.cn/en/article/doi/10.7652/xjtuxb201804006/ (snippet) | Consistent (magnitude). The R1 seed's Springer paper (5.7 C with reduced flow) was not located |
| Abrupt faults (M2) | None | 30%; residual 15 to 25 C over 1 to 4 h | None | | **Engineering assumption, no published benchmark** |
| Housing response to a cooling fault (M1) | None | 0.5 to 1.0 x thermistor residual, tau 3 h | Cooled housing sits near ambient; front bearing about 20 C, rear about 30 C at steady state (snippet) | czasopisma.pan.pl 2023 study, https://czasopisma.pan.pl/Content/131249 (snippet) | **Engineering assumption, no published benchmark** (grounded in the cooled heat path) |
| Chiller fault indication at the controller | None | No fault contact from the spindle chiller is wired to the controller: an abrupt cooler fault is detected only through the spindle thermistor (46 C overheat limit) and the spindle sensor temperature; at low spindle speed the thermistor can settle below the limit (run07: HMC-04 8 days at 42.9 C at 3,350 rpm) | None | | Shop record (engineering assumption, run07 checkpoint (a), 2026-10-29) |
| Chiller repair | 4 to 12 h | 1 to 3 days with a spare or rental unit | None | | Engineering assumption |
| Overheat limit | 46 C | Unchanged | | | Shop record (controller setting) |

## Compressors and compressed air

| Parameter | Current | New | Published range | Source | Verdict |
|---|---|---|---|---|---|
| Demand per unit, first shift average and peak, second, Saturday | 0.75, 0.65, 0.50 | 1.5 average and 2.0 peak; 1.3; 1.0 | 4 to 5 cfm per hp at 100 psig, so a 75 hp unit gives about 280 to 375 cfm; leaks under 10% well maintained, about 20% typical, 20 to 30% poorly maintained | CR4 forum rules of thumb (snippet), https://cr4.globalspec.com/thread/5082/Air-Compressor-Rules-of-Thumb ; Compressed Air Challenge Fact Sheet 7 (2026), https://www.compressedairchallenge.org/data/sites/1/media/library/factsheets2026/fact-sheet-7-compressed-air-system-leaks_updated2026.pdf | Supported: 280 cfm per unit sits at the low end; 15% leaks sits between well maintained and typical |
| N+1, four units | Duty and standby | Lead rotation over four, up to three running | | | Shop record |
| Discharge warning and trip | 100 and 110 C | Unchanged | Kaeser ASK 28: the machine shuts down at an airend discharge temperature of at most 110 C. Atlas Copco GA30: shutdown warning nominally 110 C, shutdown 120 C. One other controller (ES4000) shuts down at 117 C | Kaeser ASK 28 manual, p. 79, opened 2026-10-29: https://www.manualslib.com/manual/1775922/Kaeser-Ask-28.html?page=79 ; Atlas Copco GA30 instruction book, https://www.e-pneumatic.com/media/pdf/atlas-copco-ga-30-manual.pdf (downloaded 2026-10-29, text not extractable here; values as quoted at the pilot04 review, not verified by Claude Code); ES4000 (snippet), https://www.scribd.com/document/577576639/ES4000-basic-2 | Supported by the Kaeser setting (trip 110 C); Atlas Copco nominal settings are 10 C higher |
| Fouling onset | 1.0 duty, 0.5 standby per unit-year | 3.1e-4 per running hour | None | | **Engineering assumption, no published benchmark** |
| Airend lube event | 1 a year, duty unit | 3.1e-4 per running hour | None | | **Engineering assumption, no published benchmark** |
| Other compressor stoppages | 0.7 per duty running year | 2.17e-4 per running hour | None | | **Engineering assumption, no published benchmark** |
| Imbalance, misalignment, looseness rates | 0.18, 0.10, 0.05 per unit-year | Unchanged | None | | **Engineering assumption, no published benchmark** |

## Climate and sensor temperature

| Parameter | Current | New | Published range | Source | Verdict |
|---|---|---|---|---|---|
| Compressor room seasonal mean and amplitude, daily amplitude | 24 C, 11 C, 3 C | 22 C, 10 C, 3.5 C | Chicago O'Hare 1991 to 2020 normals: January -3.8 C, July 24.1 C, half-amplitude about 14 C outdoors | NWS Chicago normals, https://www.weather.gov/lot/Chicago_normals | Consistent: a ventilated room damps the outdoor swing. The amplitude itself is an engineering assumption |
| Heat spell | 3 to 6 C, 3 to 7 days, June to August | Unchanged | | | **Engineering assumption, no published benchmark** |
| Shop floor | 22 C, amplitude 4 C | 21.5 C, amplitude 1.75 C, no daily cycle | No shop-floor benchmark; 20 C is the ISO 1 measurement reference | Cutting Tool Engineering, https://ctemag.com/?p=96961 | Consistent. The R1 seed's CTE article on holding shops at a set temperature gave no figure |
| Running rise: spindle, pump, compressor motor, airend | 10, 15, 25, 35 C | 5, 15, 27.5, 35 C | Front bearing near 20 C and rear near 30 C on a cooled motorized spindle (snippet) | czasopisma.pan.pl (as above) | **Engineering assumption, no published benchmark** |
| Housing lag | None | First order, tau 3 h | None | | **Engineering assumption, no published benchmark** |
| Compressor housing load term | None | (0.7 + 0.3 x load fraction), mean-preserving | None | | Engineering assumption |
| Airend rise from fouling | None | 0.5 x discharge residual | None | | Engineering assumption |
| Vendor temperature limits | None | Spindle 42 and 52 C; pump 52 and 62 C; compressor motor 85 and 95 C; airend 92 and 102 C | Fixed per class and set by the user (vendor practice) | Banner VIBE-IQ technical note, https://info.bannerengineering.com/cs/groups/public/documents/literature/b_51937365.pdf | Shop record (vendor settings) |

## Sensors and the platform

| Parameter | Current | New | Published range | Source | Verdict |
|---|---|---|---|---|---|
| Hourly scalars, spectra weekly and on alert | Unchanged | Unchanged | Emerson: 13 scalars hourly, 4 spectra a day. SKF IMx-1: every few hours, waveform weekly or on alert. ABB WiMon: RMS hourly, waveform daily | Emerson AMS wireless vibration monitor, https://www.emerson.co.jp/content/dam/emerson/en/systems-and-software-syss/ams/product-data-sheets/documents/ams-wireless-vibration-monitor.pdf ; SKF (snippet), https://www.albeco.com.pl/pl/katalog/preview/SKF_Enlight_Collect.pdf ; ABB (snippet), https://library.e.abb.com/public/4da3cd9d94e54dff90f60297bbce5c71/TV0015C%20WiMon%20100%20Datasheet.pdf | Supported |
| Battery life | 14 to 20 months | 24 to 42 months | Emerson 3 to 5 years; SKF about 4 years; ABB over 5 years; KCF 2 to 8 years by interval | As above; KCF SD-VSN-3 (snippet), https://kcftech.com/wp-content/uploads/2024/10/SD-VSN-3-Data-Sheet-E-1.pdf | Supported: the low end, suited to hourly bursts |
| Re-baseline after maintenance | After install, remount and repair | Unchanged | "After heavy preventative maintenance or asset replacement" | Banner (as above) | Supported |
| Gateway dropouts | 8 per gateway-year, 1 to 12 h | Unchanged | Vendor claims over 99% delivery only | Emerson 1410 gateway, https://www.emerson.com/en-us/catalog/emerson-sku-1410-wireless-gateway (snippet) | **Engineering assumption, no published benchmark** |
| Sensor remounts | 2 for 20 points | 4 for 40 points | | | Shop record |

## CMMS, staffing and production

| Parameter | Current | New | Published range | Source | Verdict |
|---|---|---|---|---|---|
| Batch close | Flat 0 to 2 days | 65% closed in an end-of-week batch, 0 to 5 working days; the rest 0 to 2 days | 93% of 21,107 HVAC work orders closed on Fridays; openings about 20% per weekday | Conte et al., PHM Society 2021, https://papers.phmsociety.org/index.php/phme/article/download/2814/1781 | Supported (a lower share than the published case) |
| Missing failure codes | 0.10 | 0.15 | "Frequently" missing or incorrect; no share given | Hodkiewicz and Ho 2016, https://www.emerald.com/insight/content/doi/10.1108/JQME-04-2015-0013/full/html | Consistent; the share is an engineering assumption |
| Wrong codes | 0.12 | Unchanged | As above | As above | Consistent |
| Cutting utilization over scheduled time | HMC 0.72, VMC 0.55, lathe 0.55, mill-turn 0.60, grinder 0.40 | 0.72, 0.45, 0.45, 0.50, 0.40 | Spindle uptime: Top Shops median 72%, others 60%. Measured averages 26 to 29% of all hours (MachineMetrics, via a blog). No split by machine type | Modern Machine Shop Top Shops 2016, https://mmsonline.com/articles/2016-bechmarking-executive-summary-now-online ; Harmoni (snippet), https://harmoni.io/feeds/blog/machine-utilization | Consistent: the bottleneck HMC cell at the Top Shops median and other classes below the general figure. The split by class is an engineering assumption |
| Technicians | 2 | 3 | | | Shop record |
| PM compliance, repeat-alert inspection | 70%; 0.20 within 30 days | Unchanged | None | | Shop record |
| Other stoppages and controller alarm rates | Section 8.8 rates | Unchanged | No transferable CNC MTBF or alarm rate; one gear grinder study gives MTBF 909 h | IEEE Access 2022, https://doaj.org/article/51fdbae2d1e1420ca8b4a2190d29af24 (snippet) | **Engineering assumption, no published benchmark** |
| Axis and ball screw wear events | 3 to 5 in the record | 6 to 10 | | | Engineering assumption (scaled with the fleet) |

## R1 seed items not confirmed

| Seed claim | What was found |
|---|---|
| Vibration detection 1 to 9 months ahead | Not found. Only the 3 to 6 months for outer-race envelope warning is confirmed |
| ITC: stage 3 5 to 15 C, stage 4 over 25 C | Not found for bearings. The ITC and NETA delta-T tiers are for electrical equipment, and the Infraspection Institute advises against temperature-based time-to-failure |
| Spindle design life 20,000 to 30,000 h (Autodesk/ISO table) | Not found. Snippets give 10,000 to 20,000 h and about 8,000 h |
| Springer motorized-spindle paper, 5.7 C | Not located. The closest is MDPI Applied Sciences 2020, up to 6.7 C |
| Atlas Copco GA and Kaeser warning and trip settings | Resolved at the pilot04 review: Kaeser ASK 28 shuts down at most 110 C (opened); Atlas Copco GA30 warning 110 C and shutdown 120 C (book downloaded, text not extractable here) |
| CTE shop temperature control figure | Article gives no figure; only the ISO 20 C reference |
| Ingersoll Rand 16,000 h | A community procedure only, citing no manual |
