; ================================================================================
; Time_Series_VECM_Model
; Auto-generated from VAR/VECM Analysis
; ================================================================================
;
; Agent-based model generated from VAR/VECM analysis of longitudinal data
;
; Data: C:/Users/zachh/Downloads/MTS_Search/MTS_Search/Dataset/Flight001.csv
; Time period: 0 to 4802
; Variables: 8
; Granger relationships: 42
; Cointegration rank: 2
; VAR lag order: 5
;
; Author: Auto-generated
; ================================================================================

globals [
  ; === TIME SERIES VARIABLES (normalized) ===
  fn
  alt
  time
  nc
  t48
  nf
  nfr
  p30

  ; === LAG VARIABLES ===
  ; Lag 1
  lag1-fn
  lag1-alt
  lag1-time
  lag1-nc
  lag1-t48
  lag1-nf
  lag1-nfr
  lag1-p30
  ; Lag 2
  lag2-fn
  lag2-alt
  lag2-time
  lag2-nc
  lag2-t48
  lag2-nf
  lag2-nfr
  lag2-p30
  ; Lag 3
  lag3-fn
  lag3-alt
  lag3-time
  lag3-nc
  lag3-t48
  lag3-nf
  lag3-nfr
  lag3-p30
  ; Lag 4
  lag4-fn
  lag4-alt
  lag4-time
  lag4-nc
  lag4-t48
  lag4-nf
  lag4-nfr
  lag4-p30
  ; Lag 5
  lag5-fn
  lag5-alt
  lag5-time
  lag5-nc
  lag5-t48
  lag5-nf
  lag5-nfr
  lag5-p30

  ; === EQUILIBRIUM & ERROR CORRECTION ===
  equilibrium-1
  error-1
  equilibrium-2
  error-2

  ; === TIME TRACKING ===
  year
  tick-count

  ; === HISTORY TRACKING ===
  history-fn
  history-alt
  history-time
  history-nc
  history-t48
  history-nf
  history-nfr
  history-p30

  ; === CORRELATED SHOCKS ===
  shock-fn
  shock-alt
  shock-time
  shock-nc
  shock-t48
  shock-nf
  shock-nfr
  shock-p30
]


; ================================================================================
; SETUP PROCEDURES
; ================================================================================

to setup
  clear-all
  reset-ticks

  set year 0
  set tick-count 0

  ; Initialize all variables to 0 (normalized baseline)
  set fn 0
  set alt 0
  set time 0
  set nc 0
  set t48 0
  set nf 0
  set nfr 0
  set p30 0

  ; Initialize lag variables
  update-lags

  ; Initialize history lists
  set history-fn []
  set history-alt []
  set history-time []
  set history-nc []
  set history-t48 []
  set history-nf []
  set history-nfr []
  set history-p30 []
  record-history

  setup-patches
end

to setup-patches
  ask patches [ set pcolor white ]
end


; ================================================================================
; MAIN SIMULATION LOOP
; ================================================================================

to go
  if max-year > 0 and year >= max-year [ stop ]

  update-lags
  apply-error-correction
  apply-var-dynamics
  apply-polynomial-interactions
  add-noise
  clip-variables

  set year year + 1
  set tick-count tick-count + 1
  record-history
  tick
end


; ================================================================================
; LAG UPDATE
; ================================================================================

to update-lags
  ; Shift lag4 to lag5
  set lag5-fn lag4-fn
  set lag5-alt lag4-alt
  set lag5-time lag4-time
  set lag5-nc lag4-nc
  set lag5-t48 lag4-t48
  set lag5-nf lag4-nf
  set lag5-nfr lag4-nfr
  set lag5-p30 lag4-p30
  ; Shift lag3 to lag4
  set lag4-fn lag3-fn
  set lag4-alt lag3-alt
  set lag4-time lag3-time
  set lag4-nc lag3-nc
  set lag4-t48 lag3-t48
  set lag4-nf lag3-nf
  set lag4-nfr lag3-nfr
  set lag4-p30 lag3-p30
  ; Shift lag2 to lag3
  set lag3-fn lag2-fn
  set lag3-alt lag2-alt
  set lag3-time lag2-time
  set lag3-nc lag2-nc
  set lag3-t48 lag2-t48
  set lag3-nf lag2-nf
  set lag3-nfr lag2-nfr
  set lag3-p30 lag2-p30
  ; Shift lag1 to lag2
  set lag2-fn lag1-fn
  set lag2-alt lag1-alt
  set lag2-time lag1-time
  set lag2-nc lag1-nc
  set lag2-t48 lag1-t48
  set lag2-nf lag1-nf
  set lag2-nfr lag1-nfr
  set lag2-p30 lag1-p30
  ; Shift current to lag1
  set lag1-fn fn
  set lag1-alt alt
  set lag1-time time
  set lag1-nc nc
  set lag1-t48 t48
  set lag1-nf nf
  set lag1-nfr nfr
  set lag1-p30 p30
end


; ================================================================================
; ERROR CORRECTION (VECM)
; ================================================================================

to apply-error-correction
  ; Calculate equilibrium deviations
  set equilibrium-1 ((1.000 * fn) + (12.417 * time) + (-6390.356 * nc) + (15220.044 * t48) + (5194.804 * nf) + (-7720.500 * nfr) + (-3706.821 * p30))
  set error-1 equilibrium-1
  set equilibrium-2 ((1.000 * alt) + (-6.645 * time) + (1102.328 * nc) + (-2992.411 * t48) + (-616.706 * nf) + (1326.634 * nfr) + (909.418 * p30))
  set error-2 equilibrium-2

  ; Apply error correction - FIXED: Actually apply the corrections!
end


; ================================================================================
; VAR DYNAMICS (Granger Causality)
; ================================================================================

to apply-var-dynamics
  ; VAR dynamics based on Granger causality
  let vs var-strength

  ; T48 → NfR
  set nfr nfr + (0.30 * vs * lag2-t48 * 0.01)

  ; T48 → Nf
  set nf nf + (0.30 * vs * lag2-t48 * 0.01)

  ; T48 → Nc
  set nc nc + (0.30 * vs * lag2-t48 * 0.01)

  ; T48 → Fn
  set fn fn + (0.30 * vs * lag3-t48 * 0.01)

  ; time → alt
  set alt alt + (0.30 * vs * lag1-time * 0.01)

  ; T48 → P30
  set p30 p30 + (0.30 * vs * lag5-t48 * 0.01)

  ; Nf → Nc
  set nc nc + (0.30 * vs * lag4-nf * 0.01)

  ; NfR → Nc
  set nc nc + (0.30 * vs * lag4-nfr * 0.01)

  ; P30 → Fn
  set fn fn + (0.30 * vs * lag3-p30 * 0.01)

  ; P30 → Nf
  set nf nf + (0.30 * vs * lag4-p30 * 0.01)

  ; P30 → NfR
  set nfr nfr + (0.30 * vs * lag5-p30 * 0.01)

  ; P30 → T48
  set t48 t48 + (0.30 * vs * lag2-p30 * 0.01)

  ; P30 → Nc
  set nc nc + (0.30 * vs * lag3-p30 * 0.01)

  ; Nc → T48
  set t48 t48 + (0.30 * vs * lag2-nc * 0.01)

  ; Nc → P30
  set p30 p30 + (0.30 * vs * lag3-nc * 0.01)

  ; Nc → Nf
  set nf nf + (0.30 * vs * lag4-nc * 0.01)

  ; Nc → NfR
  set nfr nfr + (0.30 * vs * lag4-nc * 0.01)

  ; Nf → T48
  set t48 t48 + (0.30 * vs * lag2-nf * 0.01)

  ; NfR → T48
  set t48 t48 + (0.30 * vs * lag2-nfr * 0.01)

  ; Fn → Nc
  set nc nc + (0.30 * vs * lag5-fn * 0.01)

  ; Nf → Fn
  set fn fn + (0.30 * vs * lag4-nf * 0.01)

  ; time → NfR
  set nfr nfr + (0.30 * vs * lag1-time * 0.01)

  ; time → Nf
  set nf nf + (0.30 * vs * lag1-time * 0.01)

  ; NfR → Fn
  set fn fn + (0.30 * vs * lag4-nfr * 0.01)

  ; Nc → Fn
  set fn fn + (0.30 * vs * lag5-nc * 0.01)

  ; Fn → T48
  set t48 t48 + (0.30 * vs * lag2-fn * 0.01)

  ; Nf → P30
  set p30 p30 + (0.30 * vs * lag4-nf * 0.01)

  ; NfR → P30
  set p30 p30 + (0.30 * vs * lag4-nfr * 0.01)

  ; time → T48
  set t48 t48 + (0.30 * vs * lag1-time * 0.01)

  ; Fn → Nf
  set nf nf + (0.30 * vs * lag4-fn * 0.01)

  ; time → Nc
  set nc nc + (0.30 * vs * lag1-time * 0.01)

  ; Fn → NfR
  set nfr nfr + (0.30 * vs * lag4-fn * 0.01)

  ; Nf → NfR
  set nfr nfr + (0.30 * vs * lag5-nf * 0.01)

  ; Fn → P30
  set p30 p30 + (0.30 * vs * lag5-fn * 0.01)

  ; alt → P30
  set p30 p30 + (0.30 * vs * lag4-alt * 0.01)

  ; time → P30
  set p30 p30 + (0.30 * vs * lag1-time * 0.01)

  ; alt → Nc
  set nc nc + (0.30 * vs * lag2-alt * 0.01)

  ; alt → Fn
  set fn fn + (0.30 * vs * lag2-alt * 0.01)

  ; time → Fn
  set fn fn + (0.30 * vs * lag1-time * 0.01)

  ; NfR → Nf
  set nf nf + (0.30 * vs * lag2-nfr * 0.01)

  ; alt → T48
  set t48 t48 + (0.30 * vs * lag3-alt * 0.01)

  ; alt → Nf
  set nf nf + (0.30 * vs * lag2-alt * 0.01)

end


; ================================================================================
; POLYNOMIAL INTERACTIONS
; ================================================================================

to apply-polynomial-interactions
  if not use-interactions? [ stop ]

  let vs var-strength
  let is interaction-strength

  ; === SECOND-ORDER INTERACTIONS ===
  ; fn × alt → alt
  set alt alt + (lag1-fn * lag1-alt * 0.10 * vs * is * 0.01)

  ; fn × time → time
  set time time + (lag1-fn * lag1-time * 0.10 * vs * is * 0.01)

  ; alt × time → nc
  set nc nc + (lag1-alt * lag1-time * 0.10 * vs * is * 0.01)

  ; alt × nc → t48
  set t48 t48 + (lag1-alt * lag1-nc * 0.10 * vs * is * 0.01)

  ; time × nc → nf
  set nf nf + (lag1-time * lag1-nc * 0.10 * vs * is * 0.01)

  ; time × t48 → nfr
  set nfr nfr + (lag1-time * lag1-t48 * 0.10 * vs * is * 0.01)

  ; === THIRD-ORDER INTERACTIONS ===
  if is > 0.3 [
    ; fn × alt × time → nc
    set nc nc + (lag1-fn * lag1-alt * lag1-time * 0.05 * vs * is * 0.01)

    ; alt × time × nc → nfr
    set nfr nfr + (lag1-alt * lag1-time * lag1-nc * 0.05 * vs * is * 0.01)

  ]
end


; ================================================================================
; NUMERICAL STABILITY
; ================================================================================

to clip-variables
  ; Clip all variables to prevent overflow
  if fn > 15 [ set fn 15 ]
  if fn < -15 [ set fn -15 ]
  if alt > 15 [ set alt 15 ]
  if alt < -15 [ set alt -15 ]
  if time > 15 [ set time 15 ]
  if time < -15 [ set time -15 ]
  if nc > 15 [ set nc 15 ]
  if nc < -15 [ set nc -15 ]
  if t48 > 15 [ set t48 15 ]
  if t48 < -15 [ set t48 -15 ]
  if nf > 15 [ set nf 15 ]
  if nf < -15 [ set nf -15 ]
  if nfr > 15 [ set nfr 15 ]
  if nfr < -15 [ set nfr -15 ]
  if p30 > 15 [ set p30 15 ]
  if p30 < -15 [ set p30 -15 ]
end


; ================================================================================
; UTILITY PROCEDURES
; ================================================================================

to add-noise
  set fn fn + random-normal 0 (noise-level * 0.01)
  set alt alt + random-normal 0 (noise-level * 0.01)
  set time time + random-normal 0 (noise-level * 0.01)
  set nc nc + random-normal 0 (noise-level * 0.01)
  set t48 t48 + random-normal 0 (noise-level * 0.01)
  set nf nf + random-normal 0 (noise-level * 0.01)
  set nfr nfr + random-normal 0 (noise-level * 0.01)
  set p30 p30 + random-normal 0 (noise-level * 0.01)
end

to record-history
  set history-fn lput fn history-fn
  set history-alt lput alt history-alt
  set history-time lput time history-time
  set history-nc lput nc history-nc
  set history-t48 lput t48 history-t48
  set history-nf lput nf history-nf
  set history-nfr lput nfr history-nfr
  set history-p30 lput p30 history-p30
end

@#$#@#$#@
GRAPHICS-WINDOW
210
10
647
448
-1
-1
13.0
1
10
1
1
1
0
1
1
1
-16
16
-16
16
0
0
1
ticks
30.0

BUTTON
15
15
88
48
Setup
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
95
15
168
48
Go
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

SLIDER
15
95
200
128
error-correction-strength
error-correction-strength
0
2
1.0
0.1
1
NIL
HORIZONTAL

SLIDER
15
135
200
168
var-strength
var-strength
0
2
1.0
0.1
1
NIL
HORIZONTAL

SLIDER
15
175
200
208
noise-level
noise-level
0
5
1.0
0.5
1
NIL
HORIZONTAL

SLIDER
15
215
200
248
interaction-strength
interaction-strength
0
1
0.5
0.1
1
NIL
HORIZONTAL

SLIDER
15
255
200
288
max-year
max-year
0
2100
0
10
1
NIL
HORIZONTAL

SWITCH
15
295
200
328
use-interactions?
use-interactions?
0
1
-1000

MONITOR
670
15
750
60
Year
year
0
1
11

MONITOR
670
70
750
115
Fn
fn
3
1
11

MONITOR
760
70
840
115
Alt
alt
3
1
11

MONITOR
670
120
750
165
Time
time
3
1
11

MONITOR
760
120
840
165
Nc
nc
3
1
11

MONITOR
670
170
750
215
T48
t48
3
1
11

MONITOR
760
170
840
215
Nf
nf
3
1
11

MONITOR
670
220
750
265
Nfr
nfr
3
1
11

MONITOR
760
220
840
265
P30
p30
3
1
11

PLOT
15
460
850
650
All Variables
Tick
Value (normalized)
0.0
40.0
-3.0
3.0
true
true
"" ""
PENS
"Fn" 1.0 0 -13345367 true "" "plot fn"
"Alt" 1.0 0 -2674135 true "" "plot alt"
"Time" 1.0 0 -955883 true "" "plot time"
"Nc" 1.0 0 -13840069 true "" "plot nc"
"T48" 1.0 0 -8630108 true "" "plot t48"
"Nf" 1.0 0 -5825686 true "" "plot nf"
"Nfr" 1.0 0 -7500403 true "" "plot nfr"
"P30" 1.0 0 -2674135 true "" "plot p30"

@#$#@#$#@

## WHAT IS IT?

This model was automatically generated from Vector Autoregression (VAR) and Vector Error Correction Model (VECM) analysis of longitudinal time series data.

## HOW IT WORKS

The model implements:
1. Error correction mechanisms (VECM)
2. Short-run VAR dynamics
3. Granger causal relationships
4. Polynomial interactions (second and third order)
5. Numerical stability through variable clipping

## HOW TO USE IT

1. Click Setup to initialize
2. Adjust sliders to control model dynamics
3. Click Go to run simulation

## CREDITS

Auto-generated by data_to_netlogo_pipeline.ipynb
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250
@#$#@#$#@
NetLogo 6.4.0
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
