# Final synthesis of the DICE FEST audit (state at Wed 21 Oct 2026 00:18 WIB)

The task context is out of date. BEST_CURRENT is not sub_v19vce. It is `submissions/sub_v19nca2pp.csv`: sha 9dc0974e…, public 0.24913, recorded in D014/D015. That file is v19vce plus the NCA members nc_lgpm (gas/EUI) and nc_lepm (elec), plus post-processing rules S1, S3 and S4 from the postproc lens.

So every post-processing gain the auditors verified is already in the current best. The rows sit 0.00001 behind rank 6 on public (0.24912) and 0.00016 behind rank 5 (0.24897). No verified item closes the gap to rank 5.

## 1. Verified, actionable improvements, ranked by honest total-score gain

### A. New gains over BEST_CURRENT
There are none that the leaderboard could resolve.
- **Swap S3 for an alternative form.** Three alternative forms of the gas soft cap were checked: the integrity lens's version (fitted on P, k=0.7), the segments lens's IMR form (k=0.08), and the tried lens's hinge G. Using any one of them instead of S3 gains -0.000004 to -0.000006 on H. Scaled to test, that is about -0.000004, below LB resolution. Not recommended: S3 is the form already confirmed on the LB.
- **Ridge on the correction weights (lambda).** Nested CV gives -0.0000027 total. The per-fold signs are mixed, and ridge shrinks the correction in the wrong direction for the denser test pool. Not recommended.

### B. Gains measured against sub_v19vce, all already deployed in ncapp/nca2pp
Each rule is listed with what it does, its honest H number, the verifiers' estimate for test, and how sure we are.
1. **S1, small-meter multiplicative shift.**
   - What: multiply predictions for M in {2,3} by (1+d23) and for M==4 by (1+d4), for gas, EUI and elec.
   - H: -0.0000458.
   - Test estimate: -0.000038 to -0.000045.
   - Confidence: the M 2-3 part is solid (t 5-6). The M=4 part is weak for EUI and elec (robust t 2.2 and 2.5); it is worth about -0.000005.
2. **S3, truncated-normal soft cap on gas and EUI.**
   - What: shrink predictions near the gas truncation at 999,765 (for EUI, the cap is that value divided by A).
   - H: -0.0000187.
   - Test estimate: -0.000011 to -0.000016. Test has fewer predictions near the cap than H (0.38 vs 0.85 per 10k above the cap).
3. **S4, hard clip to bounds implied by train.**
   - H: -0.0000038 on its own, about -0.0000002 on top of S3.
4. **Combined S1+S3+S4.**
   - H: -0.0000652. All 5 folds negative; 96-97% of split-halves negative.
   - Test estimate: -0.000054 to -0.000060.
   - Realised on public (nca to ncapp): -0.00007. This matches.

### C. The one required action: port the rules to the pipeline (reproducibility, not score)
`final/src/pipeline.py` `_assemble` still only clips at 0 (lines 169-179). The NCA members (D014) are not in the final configs either. Without these, the notebook cannot reproduce either file we will choose.

Steps:
1. **Where:** in `_assemble`, after the residual-correction loop and before the duplicate copy, add the post-processing as a config flag (for example `"postproc"` in `resid_knn.json`). With the flag off, the output must still be sha 38dd55b9.
2. **S1:** for each metric (gas, eui, elec), set `p[M∈{2,3}] *= 1+d23` and `p[M==4] *= 1+d4`.
   - d = Σ(pf·r)/Σ(pf²), computed over train rows of the group from the full-train VC fit, or taken as constants from `exp/audit/postproc_nca2/12_fullvc_smallM_nca2.json`.
   - Leave the M==1 row and the 77 duplicate rows unchanged.
3. **S3:** for gas and EUI, `p += k·(−s·φ(b)/Φ(b))` with `s = p·sqrt(a0 + b0/M)` and `b = (T − p)/s`.
   - Gas: k 0.5046, a0 0.00806, b0 0.39297, T 999764.8988.
   - EUI: k 0.6574, a0 0.00839, b0 0.38715, T 999764.8988/A.
   - Compute with logpdf/logcdf.
4. **S4:**
   - gas → clip to [5.1206·A, 999764.8988]
   - eui → clip to [5.1206, 999764.8988/A]
   - elec → clip to [5.1208·A, 149.9909·A]
5. **Reference code:** copy it exactly from `exp/audit/postproc_nca2/13_candidate_nca2.py`, which is the generator of the deployed file.
6. **Regression tests:**
   - v19nca2 config with the flag on must give sha 9dc0974e…
   - v19vce config with the flag on must give `exp/audit/postproc/sub_v19vce_pp.csv` (sha 8859f0f4…)
   - flag off must give 38dd55b9…
7. **Apply exactly one truncation step.** Do not add G, the IMR cap or the integrity shrink on top of S3. Stacking costs +0.000004 on H.

## 2. Verified blunders and their fixes
1. **Impossible gas and EUI values in sub_v19vce and sub_v19nca.** Both files have 7 gas values above 1e6 (max 1,061,409) and 8 EUI values above 1e6/A. Fixed in ncapp and nca2pp: gas max is now 990,542 and nothing exceeds a bound. No action left except the port in 1C.
2. **The notebook pipeline does not reproduce BEST_CURRENT.** This is the open blunder for the deliverables. It has three parts:
   - S1/S3/S4 are missing from `_assemble`.
   - The NCA members are missing from `final/`.
   - Three XGB members (xp2_gpgea, xp2_eui, xp2_epa) are still pending in the final run because of repeated CUDA OOM.
   
   So far 72 of 72 final-run arrays compared are byte-identical to the dev arrays.
3. **Cosmetic only:** 23 unseen rows with rest=0 carry an adoption of 7.6e-6 instead of 0 (float32 noise). The score effect is about 1e-12. An optional `np.minimum(adoption, rest_float64)` would remove it.

## 3. Checked and confirmed not to be a problem
- **Submission file:** format, id order, row count, no NaN, no BOM or CRLF, float round-trip precision, and the cache matches the raw CSV.
- **Clean ratio:** `np.round(100·b02/b01, 2)` is exact on all 429,303 train rows, including 2,372 near-ties. Do not switch to half-up rounding: it would break 403 test rows. No `b01=0` cases.
- **Adoption:**
  - All 169,008 seen-subzone rows copy the train value exactly, and all 127 subzone-constant columns are identical across train and test.
  - Unseen rows equal adopt_v2, which reproduces byte-exactly.
  - The nrow_tr feature shift is harmless.
  - The CV proxy for unseen subzones is unbiased (adversarial AUC 0.50, hypergeometric counts match).
  - No formula for the missing category exists (best ridge R² 0.36). The 0.0082 adoption cost is irreducible in practice.
- **Duplicate copy:** the 77 rows are 100% precision and recall. Wider keys are impure. Gas-only twins cannot be located. Averaging test-only duplicate predictions is worth about 0.
- **Clip at 0** never fires. No target has a usable rounding grid. Elec, EUI and per-meter ratios have no hard edges other than the gas cap.
- **Global calibration** (multiplicative, affine, isotonic, lower-tail terms) gives no gain after S1/S3.
- **Segment-level bias:** no remaining mean or slope bias by A, M, A/M, prediction level, support, unseen, zone, sub_n window, or across a 203-208 feature scan, beyond S1 and the cap.
- **Correction settings:**
  - Lambda as fitted (OLS, no intercept) is fine.
  - K=30 and WIN=30 re-tuned honestly: about 0 or worse.
  - KSHR has exactly no effect, because every row has a full neighbour pool.
  - Two- or three-level K columns are worse (+0.000003).
  - Kish effective-n shrinkage is worse.
  - Separate lambda for unseen rows or per support bin is worse.
  - Lambda is robust to the difference in support mix between H and test. H has 16.8% unseen rows, not 10%.
  - Production constants match dev.
- **Forcing gas = EUI·A** hurts (+0.000134). Per-row consistency blending is not robust.
- **Structure lens:** an auxiliary elec-decimal classifier, repeated-A features, twin-partner retrieval and the K0 fuzzy key all come out at about 0 or worse.

## 4. Refuted or unverified claims
- **Tried lens, "G is a new opportunity, never installed":** refuted as new. It is the same effect as the deployed S3; adding G on top of S3 gives +0.0000001.
- **Segments lens, -0.000024 for the IMR cap:** overstated for test. Expect about -0.000015 against vce, and it is superseded by S3.
- **Integrity lens, -0.0000256 for its truncation shrink:** overstated for test (about -0.000014). Superseded by S3, and stacking it on S3 hurts.
- **"S1 constants are universal":** false. The deltas depend on the blend and must be recomputed per member set. This was done for nca and nca2.
- **The postproc lens's M=4 t-values of 5.4 / 3.1 / 5.3:** overstated. Robust values are 4.5 / 2.2 / 2.5.
- **Submit sub_v19vce_pp or sub_v19vce_GS as a probe:** obsolete. It re-measures what ncapp already measured, and both files lack the NCA members. The only remaining use is as a hedge (section 5).
- **Ridge lambda, -0.0000027:** fold signs are mixed. Not adopted.
- **Elec soft cap at 150·A:** -0.0000013 with sd 0.0000013. Not robust.
- **Under-weighted lambda for the denser test pool, about +0.0000036:** not testable without recomputing neighbours. Too small to matter.
- **Member-level ideas** (seed bagging, RealMLP, TabPFN, a truncated-likelihood gas member, more NCA variants): not measured. Expected at most -0.00001 to -0.00002 each, at 2-3 GPU hours each, on a GPU that is already contested.
- **"Public-minus-private sd of 0.0015 makes public gaps noise":** misleading as stated. That figure is the shift in one submission's absolute score, and most of it is shared by all strong teams. The relative noise between close teams is much smaller, but still about the size of a 0.0001-0.0002 gap.

## 5. Plan for the remaining ~3 days (deadline Fri 23 Oct 23:59 WIB)

**Reality check.** Post-processing headroom is used up. Every verified item is already in BEST_CURRENT. The remaining levers are worth at most about 0.00002 each, which is about the size of private noise between us and ranks 5-6. The biggest risk to the private result is no longer the score. It is failing to reproduce the chosen files from the notebook. Priorities are therefore reproducibility first, opportunistic model gains second.

**Day 1 (Wed 21 Oct)**
1. Port S1/S3/S4 as a flag (section 1C) and run the three sha regression tests.
2. Port nc_lgpm and nc_lepm into `final/configs` and `final/src`.
3. Test NCA determinism: re-run one fold and compare bytes. If NCA is not bit-reproducible, that decides the final pick (see below).
4. Give the next free GPU slot to the final-run XGB lane (xp2_gpgea, xp2_eui, xp2_epa), ahead of new NCA variants.
   - nc_leui on CPU needs about 18 hours. Do not count on it.
   - nc_gpgea runs only if the GPU is free after the XGB lane.
   - Do not kill anything: PIDs 365018 and 433015 are running.

**Day 2 (Thu 22 Oct)**
1. Full notebook run end to end. It must write both final CSVs byte-identical to the dev files.
2. Draft the PDF writeup.
3. Adopt a new model member only if the holdout chain gives a total of -0.00001 or better, it enters the notebook deterministically, and the public score does not get worse by more than 0.00003.

**Day 3 (Fri 23 Oct)**
1. Freeze at about 18:00 WIB.
2. Clean-context subagent audit for leakage and determinism.
3. Upload the notebook and PDF, and select the two final submissions.

**Submission quota.** Spend at most one slot on a probe: submit the pipeline-built v19vce+S1/S3/S4 (equal to `exp/audit/postproc/sub_v19vce_pp.csv`, expected public about 0.24917) so hedge B has its own public reading. Do not submit IMR, G or truncation-shrink variants; the LB cannot resolve them.

**Final two submissions**
- **A = sub_v19nca2pp** (sha 9dc0974e, public 0.24913, best chain), provided the notebook reproduces it byte for byte.
- **B = v19vce + S1/S3/S4** (sha 8859f0f4, expected about 0.24917, about 0.00004 behind A in expectation).
  - Why this hedge: it removes the newest and least-proven component (the NCA members: GPU determinism and late integration) and keeps the post-processing the LB already confirmed. Kaggle scores the better of the two on private, so B matters if NCA hurts on private or cannot be reproduced.
  - sub_v19ncapp is not a useful hedge: it differs from A only by the nc_lepm elec member, worth -0.000008 on the chain.
- **If NCA cannot be made bit-reproducible:** A = v19vce + S1/S3/S4 and B = v19vce + S3/S4 only (`exp/audit/postproc/sub_v19vce_pp_s34.csv`, sha 0e2aaefc). That hedges the about 5% chance that S1, which rests on few rows, comes out positive on private.

**Expected private position.** Roughly a tie with rank 6. Rank 5 needs about 0.00016, which no verified item supplies. Only favourable private noise or an unexpectedly large member gain would get there.

Key files:
- `/root/Dicefest/submissions/sub_v19nca2pp.csv`
- `/root/Dicefest/exp/audit/postproc_nca2/13_candidate_nca2.py`
- `/root/Dicefest/exp/audit/postproc_nca2/12_fullvc_smallM_nca2.json`
- `/root/Dicefest/exp/audit/postproc/sub_v19vce_pp.csv`
- `/root/Dicefest/exp/audit/postproc/sub_v19vce_pp_s34.csv`
- `/root/Dicefest/final/src/pipeline.py`
- `/root/Dicefest/reports/DECISIONS.md` (D014, D015)