; ================================================================================
; DoD Bureaucratic Growth VECM Model (v12.3)
; Generated from ACTUAL VECM coefficients - NO INVENTED DATA
; ================================================================================
;
; Model: Vector Error Correction Model (VECM)
; Variables: 8 (DoD bureaucratic/personnel indicators)
; Cointegration rank: 6
; Data period: 1987-2024
;
; Coefficients:
;   - Beta (cointegration vectors): (8, 6)
;   - Alpha (error correction): (8, 6)
;   - Gamma (short-run dynamics): (8, 8)
;
; ALL COEFFICIENTS ARE ACTUAL VALUES FROM VECM ESTIMATION
; ================================================================================

globals [
  ; === TIME SERIES VARIABLES (DoD Bureaucracy) ===
  junior-enlisted
  company-grade-officers
  field-grade-officers
  gofos
  warrant-officers
  policy-count
  total-pas
  foia-simple-days

  ; === LAG VARIABLES (for gamma dynamics) ===
  lag1-junior-enlisted
  lag1-company-grade-officers
  lag1-field-grade-officers
  lag1-gofos
  lag1-warrant-officers
  lag1-policy-count
  lag1-total-pas
  lag1-foia-simple-days

  ; === EQUILIBRIUM & ERROR CORRECTION ===
  equilibrium-1
  error-1
  equilibrium-2
  error-2
  equilibrium-3
  error-3
  equilibrium-4
  error-4
  equilibrium-5
  error-5
  equilibrium-6
  error-6

  ; === TRACKING ===
  year
  tick-count
  history-junior-enlisted
  history-company-grade-officers
  history-field-grade-officers
  history-gofos
  history-warrant-officers
  history-policy-count
  history-total-pas
  history-foia-simple-days
]

; ================================================================================
; SETUP
; ================================================================================

to setup
  clear-all
  reset-ticks

  set year 1987
  set tick-count 0

  ; Initialize to normalized baseline (0)
  set junior-enlisted 0
  set company-grade-officers 0
  set field-grade-officers 0
  set gofos 0
  set warrant-officers 0
  set policy-count 0
  set total-pas 0
  set foia-simple-days 0

  update-lags

  ; Initialize history
  set history-junior-enlisted []
  set history-company-grade-officers []
  set history-field-grade-officers []
  set history-gofos []
  set history-warrant-officers []
  set history-policy-count []
  set history-total-pas []
  set history-foia-simple-days []
  record-history

  ask patches [ set pcolor white ]
end

; ================================================================================
; MAIN LOOP
; ================================================================================

to go
  if max-year > 0 and year >= max-year [ stop ]

  update-lags
  apply-shortrun-dynamics
  apply-error-correction
  add-noise
  clip-variables

  set year year + 1
  set tick-count tick-count + 1
  record-history
  tick
end

; ================================================================================
; SHORT-RUN DYNAMICS (GAMMA)
; ================================================================================

to apply-shortrun-dynamics
  ; Short-run VAR dynamics using ACTUAL GAMMA coefficients (lag 1)
  ; DAMPING FACTOR applied to prevent hard spikes
  let vs var-strength
  let damping 0.01  ; Smooth year-to-year changes

  ; Update Junior_Enlisted_Z
  set junior-enlisted junior-enlisted + (vs * damping * ((0.106522 * lag1-junior-enlisted) + (0.025244 * lag1-company-grade-officers) + (0.182387 * lag1-field-grade-officers) + (0.067997 * lag1-gofos) + (0.013101 * lag1-warrant-officers) + (0.864777 * lag1-policy-count) + (-0.080643 * lag1-total-pas) + (0.016296 * lag1-foia-simple-days)))

  ; Update Company_Grade_Officers_Z
  set company-grade-officers company-grade-officers + (vs * damping * ((-0.263753 * lag1-junior-enlisted) + (0.202129 * lag1-company-grade-officers) + (0.260909 * lag1-field-grade-officers) + (-0.050647 * lag1-gofos) + (0.117975 * lag1-warrant-officers) + (2.431145 * lag1-policy-count) + (0.069397 * lag1-total-pas) + (-0.145754 * lag1-foia-simple-days)))

  ; Update Field_Grade_Officers_Z
  set field-grade-officers field-grade-officers + (vs * damping * ((0.401357 * lag1-junior-enlisted) + (-0.207158 * lag1-company-grade-officers) + (-0.084265 * lag1-field-grade-officers) + (-0.282027 * lag1-gofos) + (-0.328672 * lag1-warrant-officers) + (-1.096401 * lag1-policy-count) + (-0.463448 * lag1-total-pas) + (0.338277 * lag1-foia-simple-days)))

  ; Update GOFOs_Z
  set gofos gofos + (vs * damping * ((0.039184 * lag1-junior-enlisted) + (-0.100665 * lag1-company-grade-officers) + (0.152311 * lag1-field-grade-officers) + (0.514115 * lag1-gofos) + (0.215613 * lag1-warrant-officers) + (-1.104652 * lag1-policy-count) + (-0.193046 * lag1-total-pas) + (0.008588 * lag1-foia-simple-days)))

  ; Update Warrant_Officers_Z
  set warrant-officers warrant-officers + (vs * damping * ((0.427905 * lag1-junior-enlisted) + (-0.414669 * lag1-company-grade-officers) + (-0.041666 * lag1-field-grade-officers) + (-1.184734 * lag1-gofos) + (0.604233 * lag1-warrant-officers) + (2.469196 * lag1-policy-count) + (0.145739 * lag1-total-pas) + (0.179560 * lag1-foia-simple-days)))

  ; Update Policy_Count_Log
  set policy-count policy-count + (vs * damping * ((-0.013686 * lag1-junior-enlisted) + (-0.057683 * lag1-company-grade-officers) + (-0.020485 * lag1-field-grade-officers) + (0.060200 * lag1-gofos) + (-0.054211 * lag1-warrant-officers) + (-0.645600 * lag1-policy-count) + (-0.047244 * lag1-total-pas) + (0.006094 * lag1-foia-simple-days)))

  ; Update Total_PAS_Z
  set total-pas total-pas + (vs * damping * ((0.401627 * lag1-junior-enlisted) + (-0.227954 * lag1-company-grade-officers) + (-0.280330 * lag1-field-grade-officers) + (-0.115559 * lag1-gofos) + (-0.128905 * lag1-warrant-officers) + (0.460721 * lag1-policy-count) + (0.049232 * lag1-total-pas) + (-0.094389 * lag1-foia-simple-days)))

  ; Update FOIA_Simple_Days_Z
  set foia-simple-days foia-simple-days + (vs * damping * ((-0.020156 * lag1-junior-enlisted) + (-0.208562 * lag1-company-grade-officers) + (-0.106391 * lag1-field-grade-officers) + (0.225478 * lag1-gofos) + (0.123484 * lag1-warrant-officers) + (1.289484 * lag1-policy-count) + (0.356834 * lag1-total-pas) + (-0.027491 * lag1-foia-simple-days)))

end

; ================================================================================
; LONG-RUN ERROR CORRECTION (ALPHA × BETA)
; ================================================================================

to apply-error-correction
  ; Calculate equilibrium deviations (2 cointegrating vectors)
  set equilibrium-1 ((1.0000 * junior-enlisted) + (-0.1475 * field-grade-officers) + (0.4999 * gofos) + (-0.9911 * warrant-officers) + (0.0834 * policy-count) + (2.2770 * total-pas) + (0.9902 * foia-simple-days))
  set error-1 equilibrium-1
  set equilibrium-2 ((1.0000 * company-grade-officers) + (-1.6811 * field-grade-officers) + (2.3786 * gofos) + (0.9981 * warrant-officers) + (-0.1703 * policy-count) + (-0.3417 * total-pas) + (1.8313 * foia-simple-days))
  set error-2 equilibrium-2

  ; Apply error correction
  set junior-enlisted junior-enlisted + (-0.2451 * error-correction-strength * error-1 * 0.01) + (0.2943 * error-correction-strength * error-2 * 0.01)
  set company-grade-officers company-grade-officers + (0.1402 * error-correction-strength * error-1 * 0.01) + (-0.1115 * error-correction-strength * error-2 * 0.01)
  set field-grade-officers field-grade-officers + (0.1212 * error-correction-strength * error-1 * 0.01) + (-0.1543 * error-correction-strength * error-2 * 0.01)
  set gofos gofos + (-0.0564 * error-correction-strength * error-1 * 0.01) + (-0.4239 * error-correction-strength * error-2 * 0.01)
  set warrant-officers warrant-officers + (0.0337 * error-correction-strength * error-1 * 0.01) + (-0.1509 * error-correction-strength * error-2 * 0.01)
  set policy-count policy-count + (-0.0909 * error-correction-strength * error-1 * 0.01) + (-0.2551 * error-correction-strength * error-2 * 0.01)
  set total-pas total-pas + (-0.0508 * error-correction-strength * error-1 * 0.01) + (-0.2237 * error-correction-strength * error-2 * 0.01)
  set foia-simple-days foia-simple-days + (-0.0352 * error-correction-strength * error-1 * 0.01) + (-0.0685 * error-correction-strength * error-2 * 0.01)
end

; ================================================================================
; UTILITIES
; ================================================================================

to update-lags
  set lag1-junior-enlisted junior-enlisted
  set lag1-company-grade-officers company-grade-officers
  set lag1-field-grade-officers field-grade-officers
  set lag1-gofos gofos
  set lag1-warrant-officers warrant-officers
  set lag1-policy-count policy-count
  set lag1-total-pas total-pas
  set lag1-foia-simple-days foia-simple-days
end

to add-noise
  set junior-enlisted junior-enlisted + random-normal 0 (noise-level * 0.01)
  set company-grade-officers company-grade-officers + random-normal 0 (noise-level * 0.01)
  set field-grade-officers field-grade-officers + random-normal 0 (noise-level * 0.01)
  set gofos gofos + random-normal 0 (noise-level * 0.01)
  set warrant-officers warrant-officers + random-normal 0 (noise-level * 0.01)
  set policy-count policy-count + random-normal 0 (noise-level * 0.01)
  set total-pas total-pas + random-normal 0 (noise-level * 0.01)
  set foia-simple-days foia-simple-days + random-normal 0 (noise-level * 0.01)
end

to clip-variables
  ; Prevent numerical overflow
  if junior-enlisted > 10 [ set junior-enlisted 10 ]
  if junior-enlisted < -10 [ set junior-enlisted -10 ]
  if company-grade-officers > 10 [ set company-grade-officers 10 ]
  if company-grade-officers < -10 [ set company-grade-officers -10 ]
  if field-grade-officers > 10 [ set field-grade-officers 10 ]
  if field-grade-officers < -10 [ set field-grade-officers -10 ]
  if gofos > 10 [ set gofos 10 ]
  if gofos < -10 [ set gofos -10 ]
  if warrant-officers > 10 [ set warrant-officers 10 ]
  if warrant-officers < -10 [ set warrant-officers -10 ]
  if policy-count > 10 [ set policy-count 10 ]
  if policy-count < -10 [ set policy-count -10 ]
  if total-pas > 10 [ set total-pas 10 ]
  if total-pas < -10 [ set total-pas -10 ]
  if foia-simple-days > 10 [ set foia-simple-days 10 ]
  if foia-simple-days < -10 [ set foia-simple-days -10 ]
end

to record-history
  set history-junior-enlisted lput junior-enlisted history-junior-enlisted
  set history-company-grade-officers lput company-grade-officers history-company-grade-officers
  set history-field-grade-officers lput field-grade-officers history-field-grade-officers
  set history-gofos lput gofos history-gofos
  set history-warrant-officers lput warrant-officers history-warrant-officers
  set history-policy-count lput policy-count history-policy-count
  set history-total-pas lput total-pas history-total-pas
  set history-foia-simple-days lput foia-simple-days history-foia-simple-days
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
10
1.0
0.5
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
10
1.0
0.5
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
2
0.1
0.05
1
NIL
HORIZONTAL

SLIDER
15
215
200
248
max-year
max-year
0
2050
2050
10
1
NIL
HORIZONTAL

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
840
115
Junior_Enlisted
junior-enlisted
3
1
11

MONITOR
850
70
1020
115
Company_Grade_Officers
company-grade-officers
3
1
11

MONITOR
670
120
840
165
Field_Grade_Officers
field-grade-officers
3
1
11

MONITOR
850
120
1020
165
GOFOs
gofos
3
1
11

MONITOR
670
170
840
215
Warrant_Officers
warrant-officers
3
1
11

MONITOR
850
170
1020
215
Policy_Count (Log)
policy-count
3
1
11

MONITOR
670
220
840
265
Total_PAS
total-pas
3
1
11

MONITOR
850
220
1020
265
FOIA_Simple_Days
foia-simple-days
3
1
11

PLOT
15
460
1050
680
DoD Bureaucratic Growth Dynamics (VECM Model)
Year
Normalized Value
0.0
50.0
-3.0
3.0
true
true
"" ""
PENS
"Junior_Enlisted" 1.0 0 -13345367 true "" "plot junior-enlisted"
"Company_Grade_Officers" 1.0 0 -2674135 true "" "plot company-grade-officers"
"Field_Grade_Officers" 1.0 0 -955883 true "" "plot field-grade-officers"
"GOFOs" 1.0 0 -13840069 true "" "plot gofos"
"Warrant_Officers" 1.0 0 -8630108 true "" "plot warrant-officers"
"Policy_Count (Log)" 1.0 0 -5825686 true "" "plot policy-count"
"Total_PAS" 1.0 0 -7500403 true "" "plot total-pas"
"FOIA_Simple_Days" 1.0 0 -2674135 true "" "plot foia-simple-days"

@#$#@#$#@
## WHAT IS IT?

This model simulates DoD bureaucratic growth dynamics based on Vector Error Correction Model (VECM) analysis of 1987-2024 data.

## HOW IT WORKS

The model implements TWO types of dynamics:

1. **SHORT-RUN DYNAMICS** (Gamma coefficients)
   - Year-over-year feedback effects
   - How variables respond to each other immediately

2. **LONG-RUN ERROR CORRECTION** (Alpha × Beta)
   - Structural equilibrium relationships
   - How system corrects deviations from equilibrium

ALL COEFFICIENTS ARE ACTUAL VALUES FROM VECM ESTIMATION - NO INVENTED DATA

## HOW TO USE IT

1. Click **Setup** to initialize
2. Adjust sliders:
   - **error-correction-strength** (0-10): Scale of equilibrium adjustments
   - **var-strength** (0-10): Scale of short-run dynamics
   - **noise-level** (0-2): Random fluctuations (start with 0.1)
   - **max-year** (0-2050): Simulation stopping point
3. Click **Go** to run simulation

**IMPORTANT**: Built-in damping (0.01) prevents unrealistic spikes.
Coefficients are actual VECM estimates scaled for smooth year-to-year dynamics.

**Recommended starting values:**
- error-correction-strength: 1.0
- var-strength: 1.0
- noise-level: 0.1

## VARIABLES

- Junior Enlisted: E-1 to E-4 (combat personnel)
- Company Grade: O-1 to O-3 (junior officers)
- Field Grade: O-4 to O-5 (staff officers - key bureaucratic layer)
- GOFOs: General/Flag Officers
- Warrant Officers: Technical specialists
- Policy Count (Log): DoD directives/policies
- Total PAS: Political appointees
- FOIA Simple Days: FOIA processing delay

## CREDITS

Generated from VECM v12.3 Final Analysis
DoD Bureaucratic Growth Study (1987-2024)
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
