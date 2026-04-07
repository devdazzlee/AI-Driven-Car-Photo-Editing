"""
Gemini API service for car image processing.

Uses Gemini API (gemini-3.1-flash-image-preview) for all image processing.
Single API call: remove reflections, clean floor, maintain car color, keep walls/floor intact.
"""

import base64
import io
import logging
from typing import Optional
import time
import cv2
import numpy as np
from PIL import Image, ImageOps

from app.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)

GEMINI_MODEL = "gemini-3-pro-image-preview"
REQUEST_TIMEOUT_MS = 360_000  # 6 minutes
MAX_INPUT_SIZE = 1024  # Match Gemini's 1K output size
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 10


# --- Prompts ---

ENHANCE_PROMPT = (
    "Edit this car dealership photo. You have exactly TWO jobs:\n"
    "  JOB 1 — Make the studio background (walls and ceiling) pure white.\n"
    "  JOB 2 — Remove studio light reflections from the car's painted panels and glass.\n"
    "Everything else stays pixel-identical to the input. The floor, the car, every car part — unchanged.\n\n"

    "══════════════════════════════════════════════════════════════════════\n"
    "PRIME DIRECTIVE — THE VEHICLE IS 100% UNTOUCHABLE:\n"
    "Every part of the vehicle must remain exactly as it appears in the input.\n"
    "This includes: body panels, glass, trim, wheels, lights, roof rack, roof rack frame,\n"
    "ALL CARGO ON THE ROOF RACK (pipes, tubes, ladders, lumber, tools — whatever is on the rack),\n"
    "step bars, running boards, antenna, wipers, mirrors, and every other part on the vehicle.\n"
    "Only remove objects physically attached to the BUILDING (walls, ceiling) that do NOT touch the vehicle.\n"
    "If ANY object touches the vehicle — it STAYS. No exceptions.\n"
    "══════════════════════════════════════════════════════════════════════\n\n"

    "1. COMPOSITION: Do NOT flip, mirror, rotate, zoom, or move the car. "
    "Output dimensions must be identical to input. Every part visible in input must remain visible.\n\n"

    "STEP A — LOCK THE TRUE PAINT COLOR BEFORE EDITING ANYTHING:\n\n"

    "Studio lights come from ABOVE and FRONT — they contaminate upper/front surfaces.\n"
    "The ONLY valid sampling zones are: bottom edge of door panels (lowest 10%), bottom of fenders,\n"
    "lower rocker panel, shadow side of a body crease. NEVER sample upper/front-facing areas.\n\n"

    "TRUE PAINT COLOR BY CAR COLOR:\n"
    "   - BLACK/CHARCOAL/GRAPHITE/VERY DARK: true color = near-black at bottom door edge.\n"
    "     ANY lighter area on the car = studio reflection = must be filled.\n"
    "   - DARK BROWN/DARK GREY: same — sample lowest door edge.\n"
    "   - DARK BLUE/NAVY: true color = deepest blue at panel bottom, NOT lighter blue-grey.\n"
    "   - DARK GREEN: true color = deepest green at panel bottom edge.\n"
    "   - WHITE: true color = even white away from any extra-bright blobs.\n"
    "   - GREY/SILVER METALLIC: true color = mid-tone metallic grey in bottom-third of panel.\n"
    "   - ANY OTHER COLOR: true color = richest, most saturated version at panel bottom edge.\n\n"

    "CRITICAL — LARGE REFLECTIONS ON DARK CARS:\n"
    "   Dark cars have VERY LARGE reflections (30-60% of a panel). Do NOT assume a large bright\n"
    "   zone is part of the car. If it is lighter than the bottom-edge reference → it is a reflection.\n"
    "   Fill the ENTIRE bright zone edge-to-edge. Partial fills are NOT acceptable.\n\n"

    "2. BACKGROUND — WALLS AND CEILING:\n"
    "   *** STEP ZERO — DO THIS FIRST, BEFORE ANYTHING ELSE ***\n"
    "   Look at the upper portion of this photo. You will see a LARGE GREY DOME SHAPE — a wide "
    "curved arc or oval shadow that fills much of the upper background. This is a cyclorama studio "
    "shadow. Your very first action must be: PAINT IT PURE WHITE. Every single grey pixel in the "
    "wall and ceiling area — paint it #FFFFFF (255, 255, 255). Do this before touching the car, "
    "before removing reflections, before anything else.\n"
    "   The target result looks like this: the car is parked in front of a completely plain, flat, "
    "white wall — like white copy paper behind the car. No curves, no shadows, no dome, no grey. "
    "Pure blank white behind and above the car.\n\n"
    "   AFTER PAINTING BACKGROUND WHITE — VERIFY BEFORE CONTINUING:\n"
    "   The background is complete when: pure flat #FFFFFF wall and ceiling, no grey, no dome shadow.\n"
    "   REMOVE (replace with #FFFFFF): studio lights, softboxes, light stands, ceiling rigs,\n"
    "   cables on walls/ceiling, camera stands, tripods, garage doors, door frames.\n"
    "   KEEP: everything touching the vehicle — roof rack cargo, antennas, step bars, all parts.\n"
    "   If in doubt whether something is studio equipment or vehicle cargo → KEEP IT.\n\n"

    "3. FLOOR: This image has been pre-cleaned. PRESERVE THE FLOOR EXACTLY as it appears.\n"
    "   Do NOT darken, lighten, re-clean, or change the floor in any way. Copy it faithfully.\n\n"

    "4. STUDIO REFLECTION REMOVAL — PAINTED BODY PANELS:\n\n"

    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "THE ONE RULE THAT GOVERNS EVERYTHING IN THIS SECTION:\n"
    "Studio lights shine FROM ABOVE and FROM THE FRONT.\n"
    "The BOTTOM EDGE of every painted panel (lowest 5-10% of its height) is\n"
    "always in relative shadow — it is the least contaminated zone on the panel.\n"
    "Therefore: bottom edge color = TRUE paint color for that panel.\n"
    "Anything on that panel LIGHTER than its own bottom edge = studio reflection.\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

    "THE BINARY TEST — APPLY THIS TO EVERY ZONE OF EVERY PANEL:\n"
    "   Q: Is this zone lighter than the bottom edge of this same panel?\n"
    "   YES → studio reflection → fill it with the bottom-edge color\n"
    "   NO  → natural paint → leave it\n"
    "No other judgment is needed. This test is the only test.\n\n"

    "THE 4-STEP PROCESS — REPEAT FOR EVERY PAINTED PANEL ON THE CAR:\n\n"

    "   STEP 1 — GET THE REFERENCE:\n"
    "   Look at the very bottom of the panel (the bottom 5-10% strip of its height).\n"
    "   This strip is always the darkest and most accurate zone on any panel.\n"
    "   The color of this strip = PANEL_REF. Memorize it. Examples:\n"
    "     Dark grey/graphite van  → PANEL_REF is very dark charcoal grey\n"
    "     Silver metallic truck   → PANEL_REF is mid-dark silver grey\n"
    "     Black car               → PANEL_REF is deep black\n"
    "     White car               → PANEL_REF is the base white (not extra-bright zones)\n"
    "     Navy blue car           → PANEL_REF is deep dark navy blue\n\n"

    "   STEP 2 — SCAN AND MARK:\n"
    "   Scan the entire panel from bottom to top.\n"
    "   For every zone, apply the binary test: lighter than PANEL_REF?\n"
    "   Mark every zone where the answer is YES.\n"
    "   Do not skip zones. Do not exclude zones because they look large.\n"
    "   A reflection covering 60% of a panel is still 100% a reflection.\n\n"

    "   STEP 3 — FILL EVERY MARKED ZONE:\n"
    "   Fill each marked zone with PANEL_REF.\n"
    "   Requirements:\n"
    "     → Fill the COMPLETE zone, edge to edge. No partial fills.\n"
    "     → Fill must match PANEL_REF in exact hue, saturation, and darkness.\n"
    "     → Blend naturally at the boundary of the filled zone.\n"
    "     → For metallic paint: the fill must look metallic, not flat grey.\n"
    "       Sample PANEL_REF from the bottom edge where metallic sheen is visible.\n"
    "       Recreate that metallic quality in the fill.\n\n"

    "   STEP 4 — VERIFY AND REPEAT:\n"
    "   Apply the binary test again to the entire panel after filling.\n"
    "   Any zone still lighter than PANEL_REF → still a reflection → go back to Step 3.\n"
    "   A panel is COMPLETE only when zero zones are lighter than PANEL_REF.\n\n"

    "APPLY STEPS 1-4 TO EVERY PANEL BELOW — IN ORDER — DO NOT SKIP ANY:\n\n"

    "   PANEL 1 — FRONT QUARTER PANEL (between front wheel arch and A-pillar):\n"
    "   *** MOST IMPORTANT — this panel has the largest reflections on most vehicles ***\n"
    "   On vans and trucks shot from the front, this panel is directly in the studio light path.\n"
    "   Reflections here are often VERY LARGE (covering 40-80% of the panel).\n"
    "   PANEL_REF = color at the very bottom of this panel, just above the rocker/sill.\n"
    "   Fill the ENTIRE bright zone — if it covers most of the panel, fill most of the panel.\n"
    "   After: this panel must look the same darkness as the lower rocker area below it.\n\n"

    "   PANEL 2 — HOOD (entire top surface, from front edge to windshield base):\n"
    "   PANEL_REF = color at the very front lower edge of the hood.\n"
    "   Bright zones: typically a large oval or band across the centre or upper hood.\n"
    "   Fill the entire bright zone. After: hood surface looks uniformly like PANEL_REF.\n\n"

    "   PANEL 3 — ROOF:\n"
    "   PANEL_REF = color at the rear edge of the roof (lowest-lit area).\n"
    "   Fill all bright streaks or zones across the roof surface.\n\n"

    "   PANEL 4 — FRONT DOOR:\n"
    "   PANEL_REF = color at the bottom edge of the door (just above the sill).\n"
    "   Fill all bright bands, streaks, or zones. After: door matches PANEL_REF darkness.\n\n"

    "   PANEL 5 — REAR DOOR(S):\n"
    "   Same process as front door. PANEL_REF = bottom edge of rear door.\n"
    "   Both front and rear doors should match each other in darkness after treatment.\n\n"

    "   PANEL 6 — REAR QUARTER PANEL:\n"
    "   PANEL_REF = lower edge of rear quarter. Fill the upper curve bright zone.\n\n"

    "   PANEL 7 — TAILGATE / BED COVER / REAR PANEL:\n"
    "   PANEL_REF = bottom edge of this panel. Fill all bright zones on the face.\n\n"

    "   PANEL 8 — FRONT AND REAR BUMPERS:\n"
    "   PANEL_REF = the darkest portion of the bumper face. Fill bright zones.\n\n"

    "   PANEL 9 — ALL PILLARS (A, B, C, D pillars):\n"
    "   PANEL_REF = the darkest portion of that pillar. Fill bright zones on the face.\n\n"

    "FINAL CHECK — WHOLE CAR SCAN:\n"
    "After completing all panels, look at the entire car.\n"
    "Does any panel have a zone that is visibly lighter than its own bottom edge?\n"
    "YES → that panel still has a reflection → go back and fix it.\n"
    "NO  → reflection removal is complete.\n"
    "Do not return the image until the answer is NO for every single panel.\n\n"

    "COLOR PRESERVATION RULE:\n"
    "Filling reflections must NOT change the car's overall color.\n"
    "You are only removing the overexposed brightness. The hue stays identical.\n"
    "A dark grey car stays dark grey. A silver car stays silver. A black car stays black.\n"
    "The fill color always comes from PANEL_REF of that specific panel — never invented.\n\n"

    "   GLASS SURFACES — STUDIO REFLECTION REMOVAL WITH INTERIOR PRESERVATION:\n\n"

    "   UNDERSTAND GLASS: TWO THINGS THAT LOOK DIFFERENT:\n"
    "   NATURAL GLASS TONE: The areas where you see INTO the car — dark grey, dark teal, near-black.\n"
    "     Interior shapes (seat outlines, headrests, door panels) are faintly visible as dark forms.\n"
    "     This is correct glass appearance. Preserve it.\n"
    "   STUDIO REFLECTIONS ON GLASS: Bright zones where the studio light bounced off the glass surface.\n"
    "     These look like white, light-grey, or silvery patches SITTING ON TOP of the glass.\n"
    "     The interior is NOT visible through these bright zones — they block the view through.\n"
    "     Remove every one of these.\n\n"

    "   CRITICAL DISTINCTION — WHAT THE CAR INTERIOR LOOKS LIKE THROUGH GLASS:\n"
    "   The car interior is DARK. Seats are dark fabric/leather. Headrests are dark. Door panels are dark.\n"
    "   The interior does NOT create white or bright-grey patches on glass.\n"
    "   Therefore: ANY white, near-white, or strongly light-grey patch on a car window = studio reflection.\n"
    "   There is NO EXCEPTION to this rule for side windows (door windows, quarter glass, rear window).\n"
    "   A bright white rectangular or oval area on a side window is ALWAYS a studio light reflection.\n"
    "   Do not confuse it with 'interior visible through glass' — the interior is dark, not white.\n\n"

    "   ══════════════════════════════════════════════════════════════════\n"
    "   FRONT-VIEW SHOTS — FULL WINDSHIELD REFLECTION (MOST CRITICAL CASE):\n"
    "   ══════════════════════════════════════════════════════════════════\n"
    "   When the car is photographed HEAD-ON (front view), the studio lights are directly\n"
    "   in front of the camera. This causes the ENTIRE WINDSHIELD to appear as a large\n"
    "   bright white or light-grey rectangle — the whole glass surface is one big reflection.\n"
    "   *** THIS ENTIRE BRIGHT WINDSHIELD = ONE MASSIVE STUDIO REFLECTION. REMOVE ALL OF IT. ***\n\n"
    "   HOW TO FIND THE G1 REFERENCE WHEN THE WHOLE WINDSHIELD IS BRIGHT:\n"
    "   The entire windshield may look white with no dark area visible at first glance.\n"
    "   In this case, find the reference using this priority order:\n"
    "     1. Look at the very BOTTOM CORNERS of the windshield (near the A-pillars at the base).\n"
    "        These corners receive the least direct light and are often the darkest zone.\n"
    "     2. Look at the EDGES of the windshield frame (right along the rubber seal).\n"
    "        The glass immediately bordering the frame is usually the darkest part.\n"
    "     3. Look through the windshield into the cabin — even in a heavily reflected shot,\n"
    "        you can often see the dashboard, steering wheel, or partition mesh as dark shapes.\n"
    "        Use the tone of those interior objects as your reference.\n"
    "     4. If ALL of the above look bright/white — the reference tone is dark grey (L≈40-60\n"
    "        in LAB, approximately RGB 50-80, 50-80, 50-80). Use this as the fill target.\n"
    "        This is what clean car glass looks like. Never leave the windshield white.\n\n"
    "   FRONT-VIEW WINDSHIELD RULE — ABSOLUTE:\n"
    "   If the windshield in your output is WHITE or BRIGHT GREY → the edit has FAILED.\n"
    "   A clean windshield on a front-view shot must look DARK (dark grey to near-black),\n"
    "   with interior shapes (dash, steering wheel, seats, partition) faintly perceptible.\n"
    "   If your output windshield is still bright after treatment → fill the entire windshield\n"
    "   again with the dark reference tone until it is no longer bright. Do not return until done.\n\n"

    "   STEP-BY-STEP GLASS TREATMENT — APPLY TO EVERY WINDOW INDIVIDUALLY:\n\n"

    "   Step G1 — FIND THE REFERENCE TONE FOR THIS WINDOW:\n"
    "   Find the darkest area on this specific window pane where you can see into the car interior\n"
    "   (you see dark seat shapes or dark interior panels through the glass).\n"
    "   That darkest interior-showing zone = REFERENCE TONE for this window.\n"
    "   It will be dark grey, dark teal, or near-black. Remember it.\n"
    "   IF NO DARK AREA IS VISIBLE (entire window is bright): use the bottom corner of the window\n"
    "   or the edge along the rubber seal — these are always the least-lit zones. If even those\n"
    "   are bright, use a dark grey reference (RGB ~60, 60, 60) — that is what clean glass looks like.\n\n"

    "   Step G2 — MARK EVERY BRIGHT ZONE ON THIS WINDOW:\n"
    "   Scan the entire window. Mark EVERY zone that is brighter than the G1 reference tone:\n"
    "     → Bright white or near-white patches → MARK\n"
    "     → Light grey washes covering part of the window → MARK\n"
    "     → Bright diagonal or angular streaks → MARK\n"
    "     → Any zone where the glass looks foggy, hazy, or washed-out → MARK\n"
    "     → Any zone where the interior shapes become invisible because the glass is too bright → MARK\n"
    "     → THE ENTIRE WINDSHIELD if it is mostly white/bright in a front-view shot → MARK ALL OF IT\n"
    "   Do not skip partial reflections or faint patches. Mark all of them.\n"
    "   A reflection covering 90% of the windshield is still 100% a reflection — mark all of it.\n\n"

    "   Step G3 — FILL EVERY MARKED ZONE:\n"
    "   Fill each marked zone with the G1 reference tone.\n"
    "   The filled area must: look like dark glass, match G1 in darkness and hue, blend at edges.\n"
    "   The result: the filled zone looks identical to the natural dark glass reference area.\n"
    "   For large full-windshield fills: the entire windshield should become dark grey/near-black,\n"
    "   with subtle interior shapes (dash, steering wheel) perceptible through the glass.\n\n"

    "   Step G4 — VERIFY AND REPEAT:\n"
    "   After filling: compare the entire window again to the G1 reference.\n"
    "   Is any area still brighter? → Mark it. Fill it. Repeat until the ENTIRE window matches G1.\n"
    "   The window is complete only when EVERY part of it is the same darkness as the reference.\n"
    "   For front-view shots: the windshield must NOT be white or bright grey in the output.\n\n"

    "   TREAT EVERY WINDOW — MANDATORY — DO NOT SKIP:\n"
    "   → WINDSHIELD: Apply G1–G4. All bright patches removed. Interior perceptible as dark shapes.\n"
    "     FRONT-VIEW SPECIAL: If the whole windshield is a bright white rectangle — fill ALL of it.\n"
    "     The entire windshield must be DARK in the output. No exceptions.\n"
    "   → FRONT DOOR WINDOW: Apply G1–G4. Remove all bright zones including faint grey patches.\n"
    "   → REAR DOOR WINDOW(S): Apply G1–G4. Must match front door window darkness.\n"
    "   → REAR QUARTER GLASS (small pane behind rear door): HIGH FAILURE RISK — always skipped.\n"
    "     Apply G1–G4 explicitly. After treatment: must be SAME darkness as door window beside it.\n"
    "     If it still looks 'slightly grey' compared to the door window → still has reflection → fix it.\n"
    "   → REAR WINDOW: Apply G1–G4. Broad grey wash is common here. Fill top, centre, ALL zones.\n\n"

    "   RULE: Do NOT make glass a featureless black panel. Fill to G1 reference tone — not darker.\n"
    "   G1 reference is always: dark enough to look like clean glass + interior shapes perceptible.\n\n"

    "   SIDE MIRROR FACE:\n"
    "   Mirror HOUSING (outer shell) = car body color = keep exactly unchanged.\n"
    "   Mirror FACE (reflective disc) = if blown-out white → fill with natural mid-grey tone.\n\n"

    "   GLASS FINAL CHECK — ALL MUST PASS BEFORE RETURNING:\n"
    "   [ ] Windshield — DARK (not white, not bright grey), no bright patches, interior perceptible\n"
    "   [ ] FRONT-VIEW CHECK: Is the windshield still white or very bright? → NOT DONE. Fill it dark.\n"
    "   [ ] Front door window — dark, matches reference, no bright/grey patches\n"
    "   [ ] Rear door window(s) — dark, same tone as front door window\n"
    "   [ ] Rear quarter glass — DARK, same as adjacent door window (this is most commonly missed)\n"
    "   [ ] Rear window — dark throughout, no grey wash at top or centre\n"
    "   [ ] Mirror face — not solid white\n"
    "   [ ] No window is a solid featureless black — interior shapes still perceptible\n"
    "   [ ] No window has any remaining bright patch, streak, white zone, or grey wash\n\n"

    "   SECTION 4 FINAL CHECK — DO NOT RETURN UNTIL ALL PASS:\n"
    "   [ ] Car paint color — is it the SAME color as original? Same hue, same saturation, same darkness?\n"
    "   [ ] All body panels (roof, hood, doors, fenders, quarters, tailgate, bumpers) — free of ALL\n"
    "       bright streaks, blobs, washed-out zones? Every hotspot filled with locked true paint color?\n"
    "   [ ] Windshield — dark, no bright patches, interior perceptible through glass?\n"
    "   [ ] ALL side windows (front doors, rear doors, quarter glass) — dark, no light-grey patches?\n"
    "   [ ] Rear window — dark throughout, no grey wash?\n"
    "   [ ] NO window still has a bright zone, grey wash, white patch, or diagonal light streak?\n"
    "   [ ] NO window has been made a featureless black panel — interior shapes still perceptible?\n"
    "   [ ] Mirror face — not solid white?\n"
    "   If any check fails → fix that specific item. Do not return until every check passes.\n\n"

    "   DESIGN FEATURES — NEVER TOUCH:\n"
    "   → Chrome trim strips on window frames, roof rails, door sills — keep exactly\n"
    "   → Chrome door handle levers — keep exactly\n"
    "   → Chrome/silver grille bars — keep exactly\n"
    "   → Razor-thin specular line exactly on a sharp pressed body crease — keep\n"
    "   → Mirror HOUSING color — keep exactly\n"
    "   → STEP BARS / RUNNING BOARDS — the horizontal bars/steps below the doors — keep exactly\n"
    "   → Mud flaps, rear reflectors, any trim at the base of the vehicle — keep exactly\n"
    "   → All components at or near the door sill / rocker panel area — keep exactly\n\n"

    "5. CAR COLOR: Preserve exact hue, saturation, and darkness. "
    "Reflection fills must use PANEL_REF color sampled from the darkest area of that panel. "
    "Never invent a color — always sample from the car itself.\n\n"

    "6. TIRES:\n"
    "   Make tires deep black and clean. Remove all dust and discoloration from rubber surface only.\n\n"

    "7. WHEELS AND RIMS:\n"
    "   Do NOT alter, distort, fill or change wheel covers, hubcaps, rims or spokes in any way. "
    "Do not fill wheels with black. Do not remove rim details or spokes. "
    "Do not add or duplicate wheel covers that are not in the original. "
    "Keep all wheel details exactly as in the original. Any wheel distortion is unacceptable. "
    "If you cannot clean tires without distorting wheels, leave wheels exactly as they are.\n\n"

    "8. COLOR ACCURACY (THIS IS CRITICAL):\n"
    "   Do not add any rainbow effects, color shifts, prismatic colors or any color distortion "
    "to any part of the image. All colors must remain true to the original. "
    "If you cannot process a specific area without causing color distortion, leave that area "
    "exactly as it is in the original.\n\n"

    "9. VIBRANCY AND SATURATION:\n"
    "   The car paint must look vibrant, rich and natural exactly like the original. "
    "Do not reduce the saturation or vibrancy of the car paint. Do not make the car "
    "look flat, dull or matte. The car should look shiny and vibrant exactly as it did "
    "in the original photo.\n\n"

    "10. LICENSE PLATE / NUMBER PLATE (ABSOLUTE RULE):\n"
    "   The license plate or dealer plate MUST remain fully visible and completely intact in the output. "
    "Do NOT remove, erase, blur, obscure, replace or alter the license plate in any way. "
    "Do NOT replace it with a blank plate, a black rectangle, or empty space. "
    "The plate must appear in exactly the same position, size, and appearance as the original. "
    "If a dealer advertising plate (e.g. BERGENCAR.COM) is on the car, keep it exactly as-is. "
    "Removing or altering the license plate is strictly forbidden.\n\n"

    "11. CAR PARTS, COMPONENTS AND BODY PANELS (THIS IS ABSOLUTELY CRITICAL):\n"
    "   RULE A — COLOR: Do NOT change the color of ANY car part or component. "
    "Headlights, tail lights, headlight surrounds, grille, grille surround, bumpers, body panels, "
    "hood, roof, doors, door handles, side mirrors, pillar trim, step bars, running boards, "
    "and every other component must remain EXACTLY the same color as in the original photo. "
    "No color changes to any car part are acceptable under any circumstances.\n\n"
    "   RULE B — STEP BARS / RUNNING BOARDS — DO NOT REMOVE (ABSOLUTE RULE):\n"
    "   Step bars (also called running boards or side steps) are horizontal bars running along the\n"
    "   bottom side of the vehicle between the wheels. On vans and trucks they appear as a dark\n"
    "   horizontal step below the door sill, positioned close to or above the floor level.\n"
    "   *** THE AI COMMONLY MISTAKES STEP BARS FOR FLOOR/BACKGROUND AND REMOVES THEM. ***\n"
    "   *** THIS IS WRONG. STEP BARS ARE PART OF THE VEHICLE. DO NOT REMOVE THEM. ***\n"
    "   → They must remain fully visible, in the exact same position as the input image.\n"
    "   → Do NOT paint over them with floor color. Do NOT erase them.\n"
    "   → Do NOT merge them into the background. They are attached to the vehicle frame.\n\n"
    "   RULE C — NO HALLUCINATION: Do NOT add, create, invent or hallucinate any car parts, badges, "
    "decorative elements, chrome surrounds, lighting elements, or any other components that are NOT "
    "visible in the original image. You must never add anything new to the car — only clean the background.\n\n"
    "   RULE D — NO MODIFICATION: Do NOT repaint, recolor, alter or modify any part of the car body, "
    "trim, or components. The car must look physically identical to the original — same parts, same colors, "
    "same finish. If a part is black in the original it must stay black. "
    "If a part is grey in the original it must stay grey. "
    "If a part is chrome in the original it must stay chrome. "
    "ONLY the background (walls and floor) and unwanted light reflections on painted panels should change. "
    "Every single car component must be pixel-identical in color and shape to the original.\n\n"

    "12. DO NOT BLUR, SMEAR OR SMUDGE THE CAR (ABSOLUTELY CRITICAL):\n"
    "   When editing the background, you MUST keep your edits strictly confined to the background "
    "pixels. Do NOT let any blurring, smearing, feathering, blending or softening from the "
    "background edit spill onto the car or any car part. Specifically:\n"
    "   - Do NOT blur the edges of the car body where it meets the background.\n"
    "   - Do NOT smear or blend any car panel detail, edge, or surface texture.\n"
    "   - Do NOT soften, feather or degrade the sharpness of any car part near the background.\n"
    "   - Do NOT let the white background colour bleed into or onto any painted car surface.\n"
    "   - The boundary between the car and the background must remain pixel-sharp — exactly as "
    "sharp and well-defined as in the original photo.\n"
    "   - Any blurring or smearing visible on the car body, fenders, doors, roof, or edges "
    "compared to the original means the edit has FAILED. The car must look as sharp and detailed "
    "as the original photograph — every edge crisp, every detail intact.\n\n"

    "13. ANTENNA, ROOF RACK AND ROOF CARGO — DO NOT REMOVE (ABSOLUTE RULE):\n\n"

    "   ANTENNA:\n"
    "   The car may have a radio antenna / mast antenna — a thin vertical or angled rod on the\n"
    "   roof or body. This is a car component — NOT a studio cable. Keep it fully intact.\n\n"

    "   ROOF RACK AND ROOF CARGO — THIS IS THE MOST CRITICAL RULE IN THIS SECTION:\n"
    "   Work vans and commercial vehicles carry cargo on their roof racks. This cargo may include:\n"
    "     → Large diameter pipes or tubes (PVC, conduit, metal pipe) lying horizontally on the rack\n"
    "     → Ladders lying flat on top of the vehicle\n"
    "     → Lumber, planks, or long materials strapped to the rack\n"
    "     → Tool cases, storage boxes, or equipment mounted on the rack\n"
    "     → Any other materials being transported by the vehicle\n"
    "   *** THIS CARGO IS NOT STUDIO EQUIPMENT. IT IS THE CUSTOMER'S CARGO. DO NOT REMOVE IT. ***\n"
    "   The test: is the object resting ON the vehicle or the vehicle's rack? → KEEP IT.\n"
    "   Is the object attached to the studio BUILDING (wall/ceiling)? → Remove it.\n"
    "   A pipe/tube/ladder on TOP of the vehicle = vehicle cargo = KEEP IT.\n"
    "   A studio light on the CEILING = studio equipment = remove it.\n"
    "   These are completely different things. Do not confuse them.\n"
    "   → The roof rack FRAME (rails, cross-bars, uprights) — KEEP exactly as in input.\n"
    "   → All cargo on the roof rack — KEEP exactly as in input.\n"
    "   → The antenna rod — KEEP exactly as in input.\n"
    "   Removing the antenna, roof rack, or any roof cargo is strictly forbidden.\n\n"

    "Return only the edited image with no text or watermarks."
)

FLOOR_CLEAN_PROMPT = (
    "Your ONLY task in this image: clean the floor. "
    "Do NOT change the car. Do NOT change the walls or background.\n\n"

    "STEP 1 — SAMPLE AND LOCK THE TRUE DRY TILE COLOR (DO THIS BEFORE ANY CLEANING):\n"
    "Look at the floor and find tiles that are clearly DRY — matte, non-shiny, non-reflective. "
    "These exist even on floors with many wet patches: look at the edges of the frame, tiles "
    "far from the car, or any tile with a flat matte appearance.\n"
    "Sample the color of 3-4 clearly-dry tiles. That is the TRUE TILE COLOR. Memorize it.\n"
    "CRITICAL INSIGHT — wet spots are LIGHTER than dry tiles:\n"
    "  - Wet/damp patches = BRIGHTER, more reflective, lighter in color than the dry tile\n"
    "  - Dry tiles = MATTE, darker, non-reflective — this is the TRUE tile color\n"
    "  - The true tile color is always the DARKER MATTE version, not the lighter shiny version\n"
    "Your TRUE TILE COLOR is the fill target. Every cleaned spot must match it exactly.\n"
    "Do not make cleaned spots lighter than the true tile color. "
    "Do not make them darker than the true tile color. Exact match only.\n\n"

    "STEP 2 — FIND ALL CONTAMINATION:\n"
    "Scan every visible floor tile in these zones WITHOUT EXCEPTION:\n"
    "  - Directly under the car body / chassis\n"
    "  - Between the front tires\n"
    "  - Between the rear tires\n"
    "  - In front of the front bumper\n"
    "  - Behind the rear bumper\n"
    "  - Along both sides of the vehicle\n"
    "  - Every other visible floor tile\n"
    "Mark each contamination type:\n"
    "  TYPE A (wet/bright): tiles LIGHTER or more reflective than the true tile color\n"
    "  TYPE B (dark stain): tiles clearly DARKER than the true tile color\n\n"

    "STEP 3 — CLEAN EACH CONTAMINATION MARK:\n"
    "For each TYPE A (bright/wet) mark:\n"
    "  - Fill it with the TRUE TILE COLOR from Step 1\n"
    "  - The result must be the same darkness as the dry tiles around it — NOT lighter\n"
    "For each TYPE B (dark stain) mark:\n"
    "  - Fill it with the TRUE TILE COLOR from Step 1\n"
    "  - The result must match the surrounding dry tiles\n"
    "For every filled area:\n"
    "  - Keep grout lines visible\n"
    "  - Keep tile texture natural\n"
    "  - Blend seamlessly with adjacent dry tiles\n\n"

    "STEP 4 — PRESERVE THE FLOOR IDENTITY (do not change any of these):\n"
    "  - Tile size: same as input\n"
    "  - Tile material and surface texture: same as input\n"
    "  - Overall floor color: same as the TRUE TILE COLOR you sampled — no lighter, no darker\n"
    "  - Grout width and color: same as input\n\n"

    "*** CRITICAL WARNING — FLOOR BRIGHTNESS / LIGHTNESS ***:\n"
    "Dealership floors come in many shades: light grey, white-grey, medium grey, dark grey, tan. "
    "You MUST output the same shade as THIS floor — not the shade you expect a 'clean' floor to be.\n"
    "  - If the dry tiles in THIS image are LIGHT GREY → output floor must be LIGHT GREY\n"
    "  - If the dry tiles in THIS image are MEDIUM GREY → output floor must be MEDIUM GREY\n"
    "  - If the dry tiles in THIS image are DARK GREY → output floor must be DARK GREY\n"
    "  - If the dry tiles in THIS image are WHITE/CREAM → output floor must be WHITE/CREAM\n"
    "DO NOT default to a dark grey/charcoal floor just because you have seen dark dealership "
    "floors in your training data. THIS floor's dry tile color is your only reference.\n"
    "A common failure mode: light grey floor becomes dark grey after cleaning. "
    "This is WRONG. If you sampled light grey in Step 1, every cleaned tile must be light grey.\n\n"

    "STEP 5 — VERIFY:\n"
    "  [ ] Overall floor color matches the dry tile reference from Step 1 — same lightness/darkness?\n"
    "  [ ] Floor is NOT darker than the original dry tiles? (most common failure — check this)\n"
    "  [ ] Floor is NOT lighter than the original dry tiles?\n"
    "  [ ] Zero wet/bright patches visible anywhere (including under the car)?\n"
    "  [ ] Zero dark stains visible anywhere?\n"
    "  [ ] Car unchanged — same color, parts, position?\n"
    "  [ ] Walls/background unchanged?\n"
    "Return the image with a clean floor at the correct original tile color, "
    "and everything else identical to the input."
)

BACKGROUND_REMOVAL_PROMPT = (
    "Remove the background from this car photo and replace it with a clean solid white background.\n\n"

    "══════════════════════════════════════════════════════════════════════\n"
    "ABSOLUTE RULE — THE VEHICLE IS UNTOUCHABLE:\n"
    "You are editing the BACKGROUND ONLY. Every single part of the vehicle must remain\n"
    "100% unchanged — body, panels, glass, trim, wheels, lights, mirrors, bumpers, and\n"
    "EVERYTHING physically attached to or resting on the vehicle.\n"
    "══════════════════════════════════════════════════════════════════════\n\n"

    "ROOF RACK AND ROOF CARGO — THIS IS THE MOST CRITICAL RULE:\n"
    "Work vans and commercial vehicles carry cargo on their roof racks. This cargo may include:\n"
    "  → Large diameter pipes or tubes (PVC, conduit, metal pipe) lying horizontally on the rack\n"
    "  → Ladders lying flat on top of the vehicle\n"
    "  → Lumber, planks, or long materials strapped to the rack\n"
    "  → Tool cases, storage boxes, or equipment mounted on the rack\n"
    "*** THIS CARGO IS NOT STUDIO EQUIPMENT. IT IS THE CUSTOMER'S CARGO. DO NOT REMOVE IT. ***\n"
    "The test: does the object rest ON the vehicle or the vehicle's rack? → KEEP IT.\n"
    "A pipe/tube/ladder on TOP of the vehicle = vehicle cargo = KEEP IT.\n"
    "Studio lights on the ceiling = studio equipment = remove them (replace with white).\n\n"

    "VEHICLE PARTS THAT MUST NEVER BE REMOVED OR ALTERED:\n"
    "  → Roof rack frame (rails, cross-bars, uprights) — KEEP exactly\n"
    "  → ALL cargo on the roof rack (pipes, ladders, tubes, equipment) — KEEP exactly\n"
    "  → Radio antenna, shark fin antenna, any mast on the roof — KEEP exactly\n"
    "  → Step bars / running boards (horizontal bars below the doors) — KEEP exactly\n"
    "  → All body panels, trim, door handles, mirrors, wipers — KEEP exactly\n"
    "  → Every part visible in the original must remain visible in the same position\n\n"

    "WHAT TO REPLACE WITH WHITE:\n"
    "  → Studio walls, ceiling, floor, studio infrastructure — replace with white\n"
    "  → Studio lights, softboxes, light stands, cables on ceiling/walls — replace with white\n"
    "  → Only objects proven to be attached to the BUILDING (not the vehicle)\n\n"

    "If there is ANY doubt whether an object is studio equipment or vehicle cargo — KEEP IT.\n\n"

    "Return only the edited image with no text or watermarks."
)

BACKGROUND_REMOVAL_TRANSPARENT_PROMPT = (
    "Remove the background from this car photo and make it fully transparent.\n\n"

    "══════════════════════════════════════════════════════════════════════\n"
    "ABSOLUTE RULE — THE VEHICLE IS UNTOUCHABLE:\n"
    "You are editing the BACKGROUND ONLY. Every single part of the vehicle must remain\n"
    "100% unchanged — body, panels, glass, trim, wheels, lights, mirrors, bumpers, and\n"
    "EVERYTHING physically attached to or resting on the vehicle.\n"
    "══════════════════════════════════════════════════════════════════════\n\n"

    "ROOF RACK AND ROOF CARGO — THIS IS THE MOST CRITICAL RULE:\n"
    "Work vans and commercial vehicles carry cargo on their roof racks. This cargo may include:\n"
    "  → Large diameter pipes or tubes (PVC, conduit, metal pipe) lying horizontally on the rack\n"
    "  → Ladders lying flat on top of the vehicle\n"
    "  → Lumber, planks, or long materials strapped to the rack\n"
    "  → Tool cases, storage boxes, or equipment mounted on the rack\n"
    "*** THIS CARGO IS NOT STUDIO EQUIPMENT. IT IS THE CUSTOMER'S CARGO. DO NOT REMOVE IT. ***\n"
    "The test: does the object rest ON the vehicle or the vehicle's rack? → KEEP IT.\n"
    "A pipe/tube/ladder on TOP of the vehicle = vehicle cargo = KEEP IT.\n"
    "Studio lights on the ceiling = studio equipment = remove them (make transparent).\n\n"

    "VEHICLE PARTS THAT MUST NEVER BE REMOVED OR ALTERED:\n"
    "  → Roof rack frame (rails, cross-bars, uprights) — KEEP exactly\n"
    "  → ALL cargo on the roof rack (pipes, ladders, tubes, equipment) — KEEP exactly\n"
    "  → Radio antenna, shark fin antenna, any mast on the roof — KEEP exactly\n"
    "  → Step bars / running boards (horizontal bars below the doors) — KEEP exactly\n"
    "  → All body panels, trim, door handles, mirrors, wipers — KEEP exactly\n"
    "  → Every part visible in the original must remain visible in the same position\n\n"

    "WHAT TO MAKE TRANSPARENT:\n"
    "  → Studio walls, ceiling, floor, studio infrastructure — make transparent\n"
    "  → Studio lights, softboxes, light stands, cables on ceiling/walls — make transparent\n"
    "  → Only objects proven to be attached to the BUILDING (not the vehicle)\n\n"

    "If there is ANY doubt whether an object is studio equipment or vehicle cargo — KEEP IT.\n\n"

    "Return only the edited image with transparent background, no text or watermarks."
)

# Gemini 3.1 Flash Image only accepts these aspect ratios
_ALLOWED_ASPECT_RATIOS = (
    "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4",
    "8:1", "9:16", "16:9", "21:9",
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _resize_for_api(pil_img: Image.Image, max_side: int = MAX_INPUT_SIZE) -> Image.Image:
    """Resize image so longest side <= max_side."""
    w, h = pil_img.size
    if max(w, h) <= max_side:
        return pil_img
    if w >= h:
        new_w, new_h = max_side, int(h * max_side / w)
    else:
        new_w, new_h = int(w * max_side / h), max_side
    return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _aspect_ratio_str(w: int, h: int) -> str:
    """Map image dimensions to nearest allowed Gemini aspect ratio."""
    if h == 0:
        return "1:1"
    actual = w / h
    best_ratio = "1:1"
    best_diff = float("inf")
    for ratio in _ALLOWED_ASPECT_RATIOS:
        num, den = map(int, ratio.split(":"))
        target = num / den
        diff = abs(actual - target)
        if diff < best_diff:
            best_diff = diff
            best_ratio = ratio
    return best_ratio


def _get_client():
    """Lazy-load Gemini client."""
    from google import genai
    api_key = GEMINI_API_KEY
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required. Set it in backend/.env")
    return genai.Client(api_key=api_key)


def _extract_image_from_response(response) -> Image.Image:
    """Extract PIL Image from Gemini API response."""
    result_bytes = None
    parts = getattr(response, "parts", None) or (
        response.candidates[0].content.parts if response.candidates else []
    )
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            data = inline.data
            result_bytes = data if isinstance(data, bytes) else base64.b64decode(data)
            break
    if result_bytes is None:
        raise RuntimeError("Gemini did not return an image")
    return Image.open(io.BytesIO(result_bytes)).convert("RGB")


def _pil_to_jpeg_bytes(pil_img: Image.Image, quality: int = 95) -> bytes:
    """Convert PIL Image to JPEG bytes."""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _call_gemini_with_retry(client, prompt: str, img_bytes: bytes, aspect: str, label: str):
    """Call Gemini API with retry logic for server errors and timeouts."""
    from google.genai import types

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("Gemini API attempt %d/%d for %s", attempt, MAX_RETRIES, label)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    top_p=1.0,
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio=aspect),
                    http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
                ),
            )
            return response
        except Exception as e:
            error_str = str(e)
            is_retryable = any(code in error_str for code in (
                "503", "500", "502", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED",
                "timed out", "ReadTimeout", "TimeoutError",
            ))
            if is_retryable and attempt < MAX_RETRIES:
                logger.warning(
                    "AI server busy (attempt %d/%d) for %s: %s — retrying in %ds...",
                    attempt, MAX_RETRIES, label, error_str, RETRY_DELAY_SECONDS,
                )
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            logger.error("Gemini API failed after %d attempts for %s: %s", attempt, label, error_str)
            raise RuntimeError(
                "Processing failed due to high server demand. Please try again in a few minutes."
            ) from e


# ---------------------------------------------------------------------------
# Post-processing checks
# ---------------------------------------------------------------------------

def _find_car_center_x(gray: np.ndarray) -> float:
    """Find horizontal center-of-mass of the darkest region (car body)."""
    threshold = np.median(gray) * 0.85
    dark_mask = (gray < threshold).astype(np.float64)
    col_weights = dark_mask.sum(axis=0)
    total = col_weights.sum()
    if total == 0:
        return 0.5
    x_coords = np.arange(gray.shape[1], dtype=np.float64)
    return (col_weights * x_coords).sum() / total / gray.shape[1]


def _is_flipped(original: Image.Image, result: Image.Image) -> bool:
    """
    Detect horizontal flip using three independent methods.

    Why the old brightness-profile method failed:
      - For centered / front-facing cars the mass center is ~0.5 in both images → method 1 always returns False.
      - The background changes dramatically (studio colours → white), so column-average brightness
        profiles correlate poorly even without flipping → method 2 gives false negatives.

    New approach — Sobel vertical-edge profiles (method 2 replacement):
      Vertical edges (door pillars, bumper edges, headlights, mirrors) are stable features
      that don't change when background lighting changes.  The column profile of |SobelX|
      captures WHERE vertical edges sit horizontally. If the image is flipped, this profile
      correlates better with its own mirror image than with the original's profile.
      This works for off-center AND centered cars because edge patterns are asymmetric
      (more door edges on one side, exhaust on one side, etc.).
    """
    size = (256, 256)
    orig_gray = np.array(original.resize(size).convert("L"), dtype=np.float32)
    res_gray  = np.array(result.resize(size).convert("L"),  dtype=np.float32)

    # --- Method 1: Car mass center-of-gravity (works for non-centered cars) ---
    orig_cx = _find_car_center_x(orig_gray.astype(np.float64))
    res_cx  = _find_car_center_x(res_gray.astype(np.float64))
    mass_flipped = False
    cx_diff = abs(orig_cx - res_cx)
    if cx_diff > 0.10:
        mirrored_cx = 1.0 - res_cx
        if abs(orig_cx - mirrored_cx) < abs(orig_cx - res_cx):
            mass_flipped = True

    # --- Method 2: Sobel vertical-edge column profile (works for centered cars too) ---
    # |SobelX| highlights vertical edges. Column-sum gives how many vertical edges are
    # at each horizontal position. This is background-independent and asymmetric per car.
    orig_sobel = np.abs(cv2.Sobel(orig_gray, cv2.CV_32F, 1, 0, ksize=3))
    res_sobel  = np.abs(cv2.Sobel(res_gray,  cv2.CV_32F, 1, 0, ksize=3))
    orig_edge_prof = orig_sobel.sum(axis=0)
    res_edge_prof  = res_sobel.sum(axis=0)
    res_edge_flip  = res_edge_prof[::-1]
    if orig_edge_prof.std() > 1e-3 and res_edge_prof.std() > 1e-3:
        normal_edge_corr = float(np.corrcoef(orig_edge_prof, res_edge_prof)[0, 1])
        flipped_edge_corr = float(np.corrcoef(orig_edge_prof, res_edge_flip)[0, 1])
        # Flipped if mirrored profile matches better by a clear margin
        edge_flipped = (flipped_edge_corr > normal_edge_corr + 0.05)
    else:
        normal_edge_corr = flipped_edge_corr = 0.0
        edge_flipped = False

    # --- Method 3: Brightness column profile (original method, kept as tiebreaker) ---
    orig_prof = orig_gray.mean(axis=0)
    res_prof  = res_gray.mean(axis=0)
    res_prof_flip = res_prof[::-1]
    if orig_prof.std() > 1e-3 and res_prof.std() > 1e-3:
        normal_bright_corr  = float(np.corrcoef(orig_prof, res_prof)[0, 1])
        flipped_bright_corr = float(np.corrcoef(orig_prof, res_prof_flip)[0, 1])
        bright_flipped = (flipped_bright_corr > normal_bright_corr + 0.05)
    else:
        normal_bright_corr = flipped_bright_corr = 0.0
        bright_flipped = False

    # Flip confirmed if at least 2 of 3 methods agree (majority vote)
    votes = sum([mass_flipped, edge_flipped, bright_flipped])
    is_flip = votes >= 2

    logger.info(
        "Flip detection: mass=%s (cx_diff=%.3f), edge=%s (corr: %.3f vs %.3f), "
        "bright=%s (corr: %.3f vs %.3f) → votes=%d → %s",
        mass_flipped, cx_diff,
        edge_flipped, normal_edge_corr, flipped_edge_corr,
        bright_flipped, normal_bright_corr, flipped_bright_corr,
        votes, "FLIP" if is_flip else "ok",
    )
    return is_flip


def _validate_composition(original: Image.Image, result: Image.Image) -> bool:
    """
    Check that Gemini kept the car in the same position and scale.

    Uses normalized 2D cross-correlation at 64×64 thumbnail scale.
    A value above 0.60 means the two images are structurally aligned.

    Why this matters for the diff composite:
      diff = gemini - original is upscaled 6× to full resolution.
      If Gemini shifted the car 5px at 1024px, that becomes a 30px shift at 6000px.
      The diff then contains bright/dark halos around every car edge.
      Applied to the full-res original those halos create a ghosted double-car.
      This check detects the misalignment before compositing so we can retry.
    """
    size = (64, 64)
    orig_arr = np.array(original.resize(size).convert("L"), dtype=np.float64)
    res_arr  = np.array(result.resize(size).convert("L"), dtype=np.float64)

    orig_norm = orig_arr - orig_arr.mean()
    res_norm  = res_arr  - res_arr.mean()
    orig_std  = orig_norm.std()
    res_std   = res_norm.std()

    if orig_std < 1e-6 or res_std < 1e-6:
        return True  # uniform image, can't measure — pass through

    correlation = float((orig_norm * res_norm).mean() / (orig_std * res_std))
    logger.info("Composition correlation: %.3f", correlation)
    return correlation > 0.60


def _check_color_accuracy(original: Image.Image, result: Image.Image, client,
                          prompt: str, img_bytes: bytes, aspect: str, label: str,
                          mask_np: Optional[np.ndarray] = None) -> Image.Image:
    """
    If car color drifted > 5%, retry Gemini once and keep the better result.

    Uses car mask pixels only — whole-image average includes background which Gemini
    changes intentionally (walls, floor), causing false positives that skip needed retries
    and miss real car color changes (white → silver) that are hidden in the average.
    """
    orig_np = np.array(original)
    res_np  = np.array(result)

    if mask_np is not None and mask_np.any():
        # Resize mask to match the 1024px images
        mh, mw = mask_np.shape
        ih, iw = orig_np.shape[:2]
        if (mh, mw) != (ih, iw):
            mask_resized = np.array(
                Image.fromarray(mask_np.astype(np.uint8) * 255, mode='L').resize((iw, ih))
            ) > 128
        else:
            mask_resized = mask_np
        orig_car = orig_np[mask_resized].astype(np.float32)
        res_car  = res_np[mask_resized].astype(np.float32)
        if orig_car.size == 0:
            orig_car = orig_np.reshape(-1, 3).astype(np.float32)
            res_car  = res_np.reshape(-1, 3).astype(np.float32)
    else:
        orig_car = orig_np.reshape(-1, 3).astype(np.float32)
        res_car  = res_np.reshape(-1, 3).astype(np.float32)

    orig_avg   = orig_car.mean(axis=0)
    res_avg    = res_car.mean(axis=0)
    color_diff = np.abs(orig_avg - res_avg).mean() / 255.0

    logger.info(
        "Car color check: orig RGB(%.0f,%.0f,%.0f) vs gemini RGB(%.0f,%.0f,%.0f) drift=%.1f%%",
        orig_avg[0], orig_avg[1], orig_avg[2],
        res_avg[0],  res_avg[1],  res_avg[2],
        color_diff * 100,
    )

    if color_diff <= 0.05:  # 5% threshold — catches white→silver (≈25% drift)
        return result

    logger.warning("Car color drift %.1f%% exceeds 5%% for %s — retrying once", color_diff * 100, label)
    try:
        retry_response = _call_gemini_with_retry(client, prompt, img_bytes, aspect, f"{label}[retry]")
        retry_pil      = _extract_image_from_response(retry_response)
        retry_np       = np.array(retry_pil)
        if mask_np is not None and mask_np.any():
            rh, rw = retry_np.shape[:2]
            if (mh, mw) != (rh, rw):
                rmask = np.array(
                    Image.fromarray(mask_np.astype(np.uint8) * 255, mode='L').resize((rw, rh))
                ) > 128
            else:
                rmask = mask_resized
            retry_car  = retry_np[rmask].astype(np.float32)
            retry_avg  = retry_car.mean(axis=0) if retry_car.size else retry_np.reshape(-1,3).astype(np.float32).mean(axis=0)
        else:
            retry_avg  = retry_np.reshape(-1, 3).astype(np.float32).mean(axis=0)
        retry_diff = np.abs(orig_avg - retry_avg).mean() / 255.0
        if retry_diff < color_diff:
            logger.info("Retry better car color (%.1f%% vs %.1f%%) for %s", retry_diff * 100, color_diff * 100, label)
            return retry_pil
        logger.info("Original result better car color (%.1f%% vs %.1f%%) for %s", color_diff * 100, retry_diff * 100, label)
    except Exception:
        logger.warning("Color retry failed for %s, using first result", label)
    return result


# Global rembg session to avoid reloading the model on every request
REMBG_SESSION = None

def _get_car_mask_rembg(pil_img: Image.Image) -> np.ndarray:
    """Get boolean mask of the car using rembg with the small 'u2netp' model."""
    global REMBG_SESSION
    try:
        from rembg import new_session, remove
        if REMBG_SESSION is None:
            # Use u2netp (4MB) instead of u2net (176MB) to avoid OOM on Railway
            logger.info("Initializing rembg session with u2netp model...")
            REMBG_SESSION = new_session("u2netp")
        
        mask_pil = remove(pil_img, only_mask=True, session=REMBG_SESSION)
        if mask_pil.mode != 'L':
            mask_pil = mask_pil.convert('L')
        return np.array(mask_pil) > 128
    except Exception as e:
        logger.error("Failed to generate RMBG mask: %s", e)
        return np.ones((pil_img.height, pil_img.width), dtype=bool)


def _get_average_color(pil_img: Image.Image, mask_np: np.ndarray) -> np.ndarray:
    """Get average RGB color of the masked area."""
    img_np = np.array(pil_img)
    if not mask_np.any():
        return np.array([0.0, 0.0, 0.0])
    return img_np[mask_np].mean(axis=0)


def _restore_car_color(original: Image.Image, gemini_result: Image.Image, mask_np: np.ndarray) -> Image.Image:
    """
    Restore original car color to Gemini's fully-processed output using LAB mean shifts.

    Gemini handles 100% of the work: reflection removal, background cleaning, everything.
    This function ONLY corrects the color drift Gemini introduces via pure mean shifts —
    no std normalization, no scaling — preserving Gemini's structure (reflections removed).

    Why mean-shift only (no std normalization):
      std normalization multiplies every pixel by orig_std/gem_std.
      For black/neutral cars this ratio is unstable (both stds are near zero)
      and amplifies noise into a visible color cast (e.g. black → brownish).
      A plain mean shift moves the whole color distribution by a fixed offset —
      stable for every car color including black, white, and grey.

    LAB channels — car area only:
      L  (brightness) : shift mean → fixes dullness, keeps Gemini's local structure
                         (reflection spots stay dark = reflections stay removed).
      A  (red↔green)  : shift mean → corrects hue drift (e.g. red→brown).
      B  (yellow↔blue): shift mean → corrects hue drift (e.g. blue cast).
      Background       : untouched — Gemini's clean background kept as-is.
    """
    orig_np   = np.array(original).astype(np.uint8)
    gemini_np = np.array(gemini_result).astype(np.uint8)

    orig_lab   = cv2.cvtColor(orig_np,   cv2.COLOR_RGB2Lab).astype(np.float32)
    gemini_lab = cv2.cvtColor(gemini_np, cv2.COLOR_RGB2Lab).astype(np.float32)

    result_lab = gemini_lab.copy()   # start with full Gemini output (background intact)

    if not mask_np.any():
        return gemini_result

    # Median is used instead of mean because reflection pixels (bright white, A≈128, B≈128)
    # are outliers in the car mask. Mean gets pulled toward neutral by those outliers,
    # causing wrong shifts for colored cars (red→brown, blue→grey, etc.).
    # Median ignores outliers — it always reflects the dominant car paint color,
    # making this correction universal across all car colors.
    for ch, name in [(0, "L"), (1, "A"), (2, "B")]:
        orig_median = np.median(orig_lab[mask_np, ch])
        gem_median  = np.median(gemini_lab[mask_np, ch])
        shift       = orig_median - gem_median
        result_lab[mask_np, ch] = np.clip(gemini_lab[mask_np, ch] + shift, 0, 255)
        logger.info("LAB %s median-shift: %+.1f  (orig=%.1f  gem=%.1f)", name, shift, orig_median, gem_median)

    result_rgb = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_Lab2RGB)
    logger.info("Car color restored via LAB median-shifts — works for all car colors")
    return Image.fromarray(result_rgb)


def _sample_floor_color(pil_img: Image.Image, car_mask_np: np.ndarray) -> tuple[int, int, int]:
    """
    Sample the average floor color from the original image.
    Floor = background pixels in the bottom 40% of the frame.
    Returns (R, G, B) as integers.
    """
    img_np = np.array(pil_img)
    ih, iw = img_np.shape[:2]
    mh, mw = car_mask_np.shape
    if (mh, mw) != (ih, iw):
        car_mask_np = _scale_mask(car_mask_np, iw, ih)
    h, w = car_mask_np.shape
    background_mask = ~car_mask_np
    floor_row_start = int(h * 0.60)
    floor_row_mask = np.zeros((h, w), dtype=bool)
    floor_row_mask[floor_row_start:, :] = True
    floor_mask = background_mask & floor_row_mask
    if not floor_mask.any():
        return (200, 200, 200)
    avg = img_np[floor_mask].mean(axis=0)
    return (int(avg[0]), int(avg[1]), int(avg[2]))


def _restore_floor_from_original(
    original: Image.Image,
    processed: Image.Image,
    car_mask_np: np.ndarray,
) -> Image.Image:
    """
    Post-process floor color correction: keep Gemini's cleaned floor, fix its color.

    Gemini cleans the floor well (removes wet spots, puddles, dirt) but reliably shifts
    the floor color/brightness. This function corrects that shift using the original as
    the color reference, while keeping Gemini's cleaned texture.

    Sampling strategy:
      - Background mask (not car) + bottom 45% of frame = floor zone
      - Centre 60% of width only (excludes backdrop curtains at left/right edges)
      - Median per channel (not mean) — median ignores puddle outliers and gives the
        true base floor color even on heavily wet concrete
      - Apply per-channel mean-shift: Gemini's floor pixels shifted by (orig_median - proc_median)
      - Only floor pixels are corrected — car and wall pixels untouched
    """
    if original.size != processed.size:
        original = original.resize(processed.size, Image.Resampling.LANCZOS)

    proc_np = np.array(processed).astype(np.float32)
    orig_np = np.array(original).astype(np.float32)
    ph, pw = proc_np.shape[:2]

    # Resize car mask to match the processed image dimensions
    mh, mw = car_mask_np.shape
    if (mh, mw) != (ph, pw):
        car_mask_np = _scale_mask(car_mask_np, pw, ph)

    h, w = car_mask_np.shape
    background_mask = ~car_mask_np

    # Floor zone: background pixels in bottom 45% of frame
    floor_row_start = int(h * 0.55)
    floor_row_mask  = np.zeros((h, w), dtype=bool)
    floor_row_mask[floor_row_start:, :] = True
    full_floor_mask = background_mask & floor_row_mask

    # Sampling zone: centre 60% width to exclude backdrop curtains at edges
    margin = int(w * 0.20)
    centre_col_mask = np.zeros((h, w), dtype=bool)
    centre_col_mask[:, margin: w - margin] = True
    sample_mask = full_floor_mask & centre_col_mask
    if sample_mask.sum() < 200:
        sample_mask = full_floor_mask  # fallback: use full floor zone
    if sample_mask.sum() < 50:
        logger.info("Floor correction: too few floor pixels, skipping")
        return processed

    orig_sample = orig_np[sample_mask]
    proc_sample = proc_np[sample_mask]

    # Median per channel — robust to wet spot outliers
    orig_median = np.array([
        float(np.median(orig_sample[:, 0])),
        float(np.median(orig_sample[:, 1])),
        float(np.median(orig_sample[:, 2])),
    ])
    proc_median = np.array([
        float(np.median(proc_sample[:, 0])),
        float(np.median(proc_sample[:, 1])),
        float(np.median(proc_sample[:, 2])),
    ])

    drift     = orig_median - proc_median
    max_drift = float(np.abs(drift).max())

    logger.info(
        "Floor correction: orig median RGB(%.0f,%.0f,%.0f) → gemini RGB(%.0f,%.0f,%.0f) "
        "drift R%+.1f G%+.1f B%+.1f",
        orig_median[0], orig_median[1], orig_median[2],
        proc_median[0], proc_median[1], proc_median[2],
        drift[0], drift[1], drift[2],
    )

    if max_drift <= 1:
        logger.info("Floor correction: drift %.1f — negligible, floor color preserved", max_drift)
        return processed

    # Apply mean-shift to all floor pixels (keeps Gemini's cleaned texture, fixes color)
    result_np = proc_np.copy()
    result_np[full_floor_mask] = np.clip(proc_np[full_floor_mask] + drift, 0, 255)

    return Image.fromarray(result_np.astype(np.uint8))


def _apply_brightness(pil_img: Image.Image, boost: float) -> Image.Image:
    """Apply brightness boost to image. boost=1.0 means no change, 1.5 means 50% brighter."""
    if boost <= 1.0:
        return pil_img
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Brightness(pil_img)
    result = enhancer.enhance(boost)
    logger.info("Applied brightness boost: %.2f", boost)
    return result


def _calculate_lab_shifts(
    original: Image.Image, gemini_result: Image.Image, mask_np: np.ndarray
) -> tuple:
    """
    Calculate LAB median shifts needed to correct Gemini's color drift.
    Returns (shift_L, shift_A, shift_B).

    Resolution-independent: calculated at 1024px but valid at any resolution because
    the shift is a scalar offset applied uniformly — not pixel-position dependent.

    Why we filter mask pixels by brightness percentile:
      The rembg car mask captures ALL car pixels: body paint, open trunk interior,
      window glass, tire rubber. When a trunk is open the dark interior (L≈5-30) is
      included. Its low L drags the median down and produces a large spurious negative
      L shift that DARKENS the entire car. Example: orig L=29 (trunk-dominated),
      gem L=60 → shift=-31 → white car body shifts from L=230 to L=199 (very dark).

      Fix: use only pixels in the 30th–80th percentile of L in the ORIGINAL car area.
        - Bottom 30% excluded: trunk interior, deep shadows, tire rubber, dark windows
        - Top 20% excluded: specular reflections, white highlights
        - Middle 50% = car body paint — the dominant, representative color

      This is robust across all car colors:
        - White car: selects body (L=180-230), skips trunk (L<50) and reflections (L>240)
        - Black car: selects body (L=20-50), skips deep shadow (L<20) and specular (L>50)
        - Any car: percentile-based so it adapts to the actual distribution
    """
    if not mask_np.any():
        return (0.0, 0.0, 0.0)
    orig_np  = np.array(original).astype(np.uint8)
    gem_np   = np.array(gemini_result).astype(np.uint8)
    orig_lab = cv2.cvtColor(orig_np, cv2.COLOR_RGB2Lab).astype(np.float32)
    gem_lab  = cv2.cvtColor(gem_np,  cv2.COLOR_RGB2Lab).astype(np.float32)

    # Build brightness-filtered mask using the original image's L distribution
    # within the car mask. Use 30th–80th percentile range → car body paint only.
    l_vals = orig_lab[mask_np, 0]
    l_lo   = float(np.percentile(l_vals, 30))
    l_hi   = float(np.percentile(l_vals, 80))
    paint_mask = mask_np & (orig_lab[:, :, 0] >= l_lo) & (orig_lab[:, :, 0] <= l_hi)

    # Fallback: if percentile filter removes too many pixels, use full mask
    if paint_mask.sum() < 50:
        paint_mask = mask_np
        logger.info("LAB shift: paint mask too sparse, falling back to full car mask")

    logger.info(
        "LAB shift sampling: %d paint pixels (L %.0f–%.0f) from %d total car pixels",
        paint_mask.sum(), l_lo, l_hi, mask_np.sum(),
    )

    shifts = []
    for ch, name in [(0, "L"), (1, "A"), (2, "B")]:
        orig_med = float(np.median(orig_lab[paint_mask, ch]))
        gem_med  = float(np.median(gem_lab[paint_mask, ch]))
        shift    = orig_med - gem_med
        shifts.append(shift)
        logger.info("LAB %s shift: %+.1f  (orig=%.1f  gem=%.1f)", name, shift, orig_med, gem_med)
    return tuple(shifts)


def _apply_lab_shifts(
    target: Image.Image, mask_np: np.ndarray, shifts: tuple
) -> Image.Image:
    """Apply pre-calculated LAB shifts to the car area of any image at any resolution."""
    if not mask_np.any() or all(abs(s) < 0.5 for s in shifts):
        return target
    t_np  = np.array(target).astype(np.uint8)
    t_lab = cv2.cvtColor(t_np, cv2.COLOR_RGB2Lab).astype(np.float32)
    for ch, shift in enumerate(shifts):
        t_lab[mask_np, ch] = np.clip(t_lab[mask_np, ch] + shift, 0, 255)
    logger.info("Applied LAB shifts %s at %s", tuple(round(s, 1) for s in shifts), target.size)
    return Image.fromarray(cv2.cvtColor(t_lab.astype(np.uint8), cv2.COLOR_Lab2RGB))


def _scale_mask(mask_np: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize a boolean mask to (target_w, target_h) with LANCZOS then threshold."""
    mask_img = Image.fromarray((mask_np.astype(np.uint8) * 255), mode='L')
    return np.array(mask_img.resize((target_w, target_h), Image.Resampling.LANCZOS)) > 128


def _force_wall_background_white(
    processed: Image.Image,
    car_mask_np: np.ndarray,
    brightness_threshold: int = 130,
    edge_buffer_px: int = 12,
) -> Image.Image:
    """
    Force all wall/ceiling background pixels to pure white (#FFFFFF).

    This is a deterministic code fix that Gemini's prompt instructions cannot reliably
    achieve because Gemini treats the cyclorama shadow and grey gradient as physically
    correct studio lighting and preserves them even when told not to.

    What this removes:
      - The cyclorama curved arc / dome shadow at the top of the background
      - Grey gradient / hollow shadow that Gemini renders on the infinity cove wall
      - Any off-white, grey, or tonal variation Gemini left in the wall/ceiling area

    Two protection layers prevent car pixels from being accidentally forced to white:

    Layer 1 — Mask dilation (edge_buffer_px=12):
      rembg slightly underestimates the car boundary — bumper corners, fender edges,
      and side panels are often 5-15px inside the true car edge. Without dilation,
      this 'gap' gets forced to white, creating a visible white halo eating into the
      car body (especially at bumper/fender corners visible against the background).
      Dilating the car mask by 12px creates a safe buffer: these boundary pixels are
      protected from being overwritten regardless of their brightness.

    Layer 2 — Brightness threshold (brightness_threshold=130):
      rembg also misses thin elements above the car body (antenna masts, roof cargo).
      Those elements are DARK, so even if they're outside the dilated mask, the
      brightness check preserves them (only light pixels get forced to white).

    The wall zone is defined as the top 58% of the frame height. The floor zone
    starts around 55-60% so there is a small safe overlap — floor correction
    (`_restore_floor_from_original`) runs after this and restores the floor correctly.
    """
    proc_np = np.array(processed)
    ph, pw = proc_np.shape[:2]

    if car_mask_np.shape != (ph, pw):
        car_mask = _scale_mask(car_mask_np, pw, ph)
    else:
        car_mask = car_mask_np

    # Dilate car mask by edge_buffer_px to create a safety buffer around car edges.
    # rembg underestimates boundaries by 5-15px; dilation ensures bumper corners
    # and fender edges are never accidentally painted white.
    if edge_buffer_px > 0:
        k = edge_buffer_px * 2 + 1
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        car_mask_dilated = cv2.dilate(
            car_mask.astype(np.uint8) * 255, dilate_kernel, iterations=1
        ) > 128
    else:
        car_mask_dilated = car_mask

    # Wall/ceiling zone: top 58% of frame rows
    wall_row_end = int(ph * 0.58)
    wall_row_mask = np.zeros((ph, pw), dtype=bool)
    wall_row_mask[:wall_row_end, :] = True

    # Non-car (using dilated mask) pixels in wall zone
    wall_bg_mask = (~car_mask_dilated) & wall_row_mask

    # Only whiten pixels that are already light-colored (actual studio background).
    # Dark pixels (< brightness_threshold) in the wall zone are likely car parts
    # (antenna masts, roof cargo, thin structures) that rembg missed — preserve them.
    pixel_brightness = proc_np.mean(axis=2)  # per-pixel mean across RGB channels
    light_pixel_mask = pixel_brightness >= brightness_threshold

    # Final mask: non-car (dilated) AND wall zone AND already light-colored
    final_mask = wall_bg_mask & light_pixel_mask

    preserved_count = int(wall_bg_mask.sum()) - int(final_mask.sum())
    result_np = proc_np.copy()
    result_np[final_mask] = [255, 255, 255]

    logger.info(
        "Wall background: forced %d pixels to white (top 58%%, buffer=%dpx, brightness>=%d); "
        "preserved %d pixels (car edge buffer + dark parts)",
        int(final_mask.sum()), edge_buffer_px, brightness_threshold, preserved_count,
    )
    return Image.fromarray(result_np)


def _sharpen_car_detail(
    processed: Image.Image,
    car_mask_np: np.ndarray,
) -> Image.Image:
    """
    Restore sharpness to the car body that Gemini softens during background editing.

    Gemini blurs and feathers car edges and panel surfaces when processing the background.
    This applies an unsharp mask strictly to the car pixels only — background pixels
    are never touched. The unsharp mask reconstructs high-frequency edge detail.

    Parameters chosen conservatively (sigma=1.0, strength=1.4) to sharpen edges
    without amplifying noise or creating halos on smooth paint surfaces.
    """
    proc_np = np.array(processed)
    ph, pw = proc_np.shape[:2]

    if car_mask_np.shape != (ph, pw):
        car_mask = _scale_mask(car_mask_np, pw, ph)
    else:
        car_mask = car_mask_np

    if not car_mask.any():
        return processed

    proc_bgr = cv2.cvtColor(proc_np, cv2.COLOR_RGB2BGR)

    # Unsharp mask: sharpened = original + (original - blurred) * amount
    blurred   = cv2.GaussianBlur(proc_bgr, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(proc_bgr, 1.4, blurred, -0.4, 0)
    sharpened_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)

    # Apply only to car area — background stays untouched
    result_np = proc_np.copy()
    result_np[car_mask] = sharpened_rgb[car_mask]

    logger.info("Car sharpening: applied unsharp mask to %d car pixels", int(car_mask.sum()))
    return Image.fromarray(result_np)


def _clean_floor_spots_inpaint(
    processed: Image.Image,
    car_mask_np: np.ndarray,
    darkness_threshold: int = 18,
) -> Image.Image:
    """
    Remove dark dirty spots and stains from the floor that Gemini left behind.

    Strategy:
      1. Identify floor zone: background (not car) pixels in the bottom 55% of the frame.
      2. Estimate the 'clean baseline' of the floor using a large median blur kernel.
         The kernel is big enough to span across individual spots/marks so the median
         represents the clean tile base colour, not the spot.
      3. Any floor pixel significantly DARKER than its local baseline is a spot/stain.
      4. Inpaint those flagged pixels with surrounding clean floor texture (TELEA method).

    darkness_threshold: how many intensity units darker than local median counts as a spot.
      Lower → more aggressive (catches faint marks). Default 18 is a good balance.
    """
    proc_np = np.array(processed)
    ph, pw = proc_np.shape[:2]

    # Resize car mask to match the processed image
    mh, mw = car_mask_np.shape
    if (mh, mw) != (ph, pw):
        car_mask = _scale_mask(car_mask_np, pw, ph)
    else:
        car_mask = car_mask_np

    # Floor zone: non-car pixels in the bottom 55% of the frame
    background_mask = ~car_mask
    floor_row_start = int(ph * 0.45)
    floor_row_mask  = np.zeros((ph, pw), dtype=bool)
    floor_row_mask[floor_row_start:, :] = True
    floor_zone = background_mask & floor_row_mask

    if not floor_zone.any():
        logger.info("Floor spot removal: no floor zone found, skipping")
        return processed

    # Convert to grayscale for spot detection
    gray = cv2.cvtColor(proc_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Large median blur kernel — spans across individual spots/stains to get clean baseline.
    # Must be odd. 81px at 1024px resolution is about 8% of frame width.
    blur_k = 81
    local_baseline = cv2.medianBlur(gray.astype(np.uint8), blur_k).astype(np.float32)

    # Spots = floor pixels that are significantly DARKER than the local clean baseline
    darkness = local_baseline - gray   # positive where pixel is darker than surroundings
    spot_mask = floor_zone & (darkness > darkness_threshold)

    spot_count = int(spot_mask.sum())
    floor_count = int(floor_zone.sum())
    if spot_count == 0:
        logger.info("Floor spot removal: no spots found (threshold=%d)", darkness_threshold)
        return processed

    logger.info(
        "Floor spot removal: %d spot pixels (%.1f%% of floor) — inpainting (threshold=%d)",
        spot_count, 100.0 * spot_count / max(floor_count, 1), darkness_threshold,
    )

    # Dilate the spot mask slightly to catch spot edges cleanly
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    spot_mask_u8 = cv2.dilate(spot_mask.astype(np.uint8) * 255, dilate_k, iterations=1)

    # TELEA inpainting: fast, excellent at reconstructing smooth/textured surfaces
    proc_bgr      = cv2.cvtColor(proc_np, cv2.COLOR_RGB2BGR)
    inpainted_bgr = cv2.inpaint(proc_bgr, spot_mask_u8, inpaintRadius=12, flags=cv2.INPAINT_TELEA)
    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

    logger.info("Floor spot removal: complete")
    return Image.fromarray(inpainted_rgb)


# ---------------------------------------------------------------------------
# Quality validation helpers
# ---------------------------------------------------------------------------

def _check_reflections_remaining(
    original: Image.Image,
    result: Image.Image,
    car_mask_np: np.ndarray,
) -> tuple[bool, float]:
    """
    Check whether studio light reflections remain on the car's painted panels.

    Strategy:
      Studio reflections make car panels BRIGHTER than they should be.
      The original image has reflections; the result should have them removed.
      We measure the 90th-percentile brightness of car pixels in both images.
      If the result's bright tail is NOT significantly reduced compared to the
      original, reflections were not properly removed.

      Additionally: we check the proportion of car pixels that are unusually
      bright relative to the car's own median (these are reflection hotspots).
      If this proportion is still high in the result → reflections remain.

    Returns:
      (reflections_remain: bool, reflection_score: float)
      reflection_score = fraction of car pixels that are hotspots (0.0–1.0).
      reflections_remain = True if score > threshold (default 0.08 = 8% of car).
    """
    if not car_mask_np.any():
        return False, 0.0

    orig_np   = np.array(original.convert("L"), dtype=np.float32)
    result_np = np.array(result.convert("L"),   dtype=np.float32)

    # Resize mask to match RESULT dimensions — result may differ from original/mask size
    rh, rw = result_np.shape[:2]
    mh, mw = car_mask_np.shape
    if (mh, mw) != (rh, rw):
        mask = _scale_mask(car_mask_np, rw, rh)
    else:
        mask = car_mask_np

    if not mask.any():
        return False, 0.0

    result_car = result_np[mask]

    # Median brightness of car pixels in result = true panel color zone
    car_median = float(np.median(result_car))

    # Hotspot threshold: pixels significantly brighter than the car median
    # Using median + 40 intensity units as the hotspot line.
    # This adapts to car color: dark cars have low median, so hotspot = any bright zone.
    # White cars have high median, so only extra-overexposed zones trigger.
    hotspot_threshold = min(car_median + 40, 230)
    hotspot_fraction  = float((result_car > hotspot_threshold).mean())

    # Reflections remain if > 8% of car pixels are hotspots
    reflections_remain = hotspot_fraction > 0.08

    logger.info(
        "Reflection check: car median=%.1f, hotspot_threshold=%.1f, "
        "hotspot_fraction=%.1f%% → %s",
        car_median, hotspot_threshold, hotspot_fraction * 100,
        "REFLECTIONS REMAIN" if reflections_remain else "ok",
    )
    return reflections_remain, hotspot_fraction


def _check_color_drift(
    original: Image.Image,
    result: Image.Image,
    car_mask_np: np.ndarray,
) -> tuple[bool, float]:
    """
    Check whether Gemini changed the car's paint color.

    Compares LAB median of the 30th–80th percentile brightness band of car pixels
    (paint body only — excludes glass, tires, dark trunk interiors, bright reflections).
    Returns (color_drifted: bool, drift_amount: float 0.0–1.0).
    Color is considered drifted if average LAB channel difference > 8 units (≈3%).
    """
    if not car_mask_np.any():
        return False, 0.0

    orig_np   = np.array(original).astype(np.uint8)
    result_np = np.array(result).astype(np.uint8)

    # Resize mask to match RESULT dimensions
    rh, rw = result_np.shape[:2]
    mh, mw = car_mask_np.shape
    if (mh, mw) != (rh, rw):
        mask = _scale_mask(car_mask_np, rw, rh)
    else:
        mask = car_mask_np

    # Also resize original to match result for consistent LAB comparison
    if orig_np.shape[:2] != (rh, rw):
        original = original.resize((rw, rh), Image.Resampling.LANCZOS)
        orig_np = np.array(original).astype(np.uint8)

    if not mask.any():
        return False, 0.0

    orig_lab   = cv2.cvtColor(orig_np,   cv2.COLOR_RGB2Lab).astype(np.float32)
    result_lab = cv2.cvtColor(result_np, cv2.COLOR_RGB2Lab).astype(np.float32)

    # Use 30th–80th percentile of original L to isolate paint body pixels
    l_vals = orig_lab[mask, 0]
    l_lo   = float(np.percentile(l_vals, 30))
    l_hi   = float(np.percentile(l_vals, 80))
    paint_mask = mask & (orig_lab[:, :, 0] >= l_lo) & (orig_lab[:, :, 0] <= l_hi)
    if paint_mask.sum() < 50:
        paint_mask = mask

    orig_med   = np.array([np.median(orig_lab[paint_mask, ch])   for ch in range(3)])
    result_med = np.array([np.median(result_lab[paint_mask, ch]) for ch in range(3)])

    # LAB units: L is 0–100, A/B are ±128 (scaled to 0–255 in OpenCV).
    # Normalise all channels to 0–1 for a single drift score.
    drift_L = abs(float(orig_med[0]) - float(result_med[0])) / 100.0
    drift_A = abs(float(orig_med[1]) - float(result_med[1])) / 255.0
    drift_B = abs(float(orig_med[2]) - float(result_med[2])) / 255.0
    drift   = (drift_L + drift_A + drift_B) / 3.0

    color_drifted = drift > 0.03   # 3% threshold across LAB channels

    logger.info(
        "Color drift check: LAB orig(%.1f,%.1f,%.1f) → result(%.1f,%.1f,%.1f) "
        "drift=%.1f%% → %s",
        orig_med[0], orig_med[1], orig_med[2],
        result_med[0], result_med[1], result_med[2],
        drift * 100,
        "COLOR DRIFTED" if color_drifted else "ok",
    )
    return color_drifted, drift


def _build_retry_prompt(reflections_remain: bool, reflection_score: float,
                        color_drifted: bool, color_drift: float) -> str:
    """
    Build a targeted retry prompt that tells Gemini exactly what failed in attempt 1.
    More specific than the general ENHANCE_PROMPT — focuses only on what needs fixing.
    """
    issues = []
    if reflections_remain:
        issues.append(
            f"ISSUE 1 — REFLECTIONS NOT REMOVED ({reflection_score*100:.0f}% of car pixels "
            f"are still bright hotspots):\n"
            "The previous attempt left studio light reflections on the car panels.\n"
            "YOU MUST:\n"
            "  → For EVERY painted panel: sample the bottom edge (lowest 5-10% of panel height).\n"
            "    That bottom-edge color = PANEL_REF = the true paint color.\n"
            "  → Fill EVERY zone on that panel that is lighter than PANEL_REF.\n"
            "  → This includes large bright zones covering most of a panel — fill all of them.\n"
            "  → After filling, no area on any panel should be brighter than its own bottom edge.\n"
            "  → Check hood, roof, front quarter panel, all doors, rear quarter, bumpers — ALL panels.\n"
            "  → For windshield and glass: fill all white/bright-grey patches with dark glass tone.\n"
            "    Front-view shots: the ENTIRE windshield may need to be filled dark.\n"
        )
    if color_drifted:
        issues.append(
            f"ISSUE 2 — CAR COLOR CHANGED ({color_drift*100:.0f}% LAB drift from original):\n"
            "The car's paint color shifted from the original. You must restore the exact color.\n"
            "YOU MUST:\n"
            "  → Sample the true paint color from the BOTTOM EDGE of a door or fender.\n"
            "  → When filling reflections, use ONLY this sampled color — never invent or lighten it.\n"
            "  → The car in the output must be the exact same hue and darkness as the input.\n"
            "  → Dark cars must stay dark. Navy blue must stay navy blue. Black must stay black.\n"
        )

    issue_text = "\n".join(issues)

    return (
        "This is attempt 2. The previous edit had quality issues that must be fixed.\n\n"
        f"{issue_text}\n"
        "EVERYTHING ELSE is correct — do NOT change the background, floor, or car position.\n"
        "Focus ONLY on fixing the issues listed above.\n\n"
        "ABSOLUTE RULE — DO NOT REMOVE ANY CAR PART:\n"
        "Every part of the vehicle must remain exactly as it appears — body, glass, trim, wheels,\n"
        "lights, roof rack, cargo, antenna, step bars, mirrors. Only fix the listed issues.\n\n"
        "Return only the corrected image with no text or watermarks."
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_car_image(
    image_data: bytes,
    filename: str,
    mode: str = "enhance-preserve",
    output_format: str = "png",
    background: str = "white",
    lighting_boost: float = 1.0,
) -> bytes:
    """
    Process car image.

    enhance-preserve: single Gemini call to clean background, remove reflections, keep floor/walls
    standard: full image to Gemini for background removal
    """
    from app.services.image_utils import load_image

    pil_img = load_image(image_data, filename)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    orig_w, orig_h = pil_img.size

    # Resize to 1024px for Gemini API
    pil_img_small = _resize_for_api(pil_img)

    w, h = pil_img_small.size
    aspect = _aspect_ratio_str(w, h)
    img_bytes = _pil_to_jpeg_bytes(pil_img_small)
    client = _get_client()

    if mode == "enhance-preserve":
        # -----------------------------------------------------------------------
        # Two-stage Gemini pipeline
        # Stage 1: Floor cleaning ONLY — focused prompt with no competing rules.
        #          One job = clean floor. Car/background unchanged.
        # Stage 2: Full edit — background white, reflections removed, car preserved.
        #          Starts with an already-clean floor so preservation rules never
        #          conflict with cleaning — the floor is already clean.
        # -----------------------------------------------------------------------
        logger.info("Stage 1/2 — floor cleaning for %s", filename)
        floor_response = _call_gemini_with_retry(
            client, FLOOR_CLEAN_PROMPT, img_bytes, aspect, f"{filename}[floor]"
        )
        floor_clean_pil = _extract_image_from_response(floor_response)
        logger.info("Stage 1 output: %s — passing to Stage 2", floor_clean_pil.size)

        # Ensure Stage 1 output matches the input size for Stage 2
        if floor_clean_pil.size != pil_img_small.size:
            floor_clean_pil = floor_clean_pil.resize(pil_img_small.size, Image.Resampling.LANCZOS)

        logger.info("Stage 2/2 — full edit for %s", filename)
        stage2_bytes = _pil_to_jpeg_bytes(floor_clean_pil)
        response = _call_gemini_with_retry(
            client, ENHANCE_PROMPT, stage2_bytes, aspect, f"{filename}[edit]"
        )
        result_pil = _extract_image_from_response(response)
        logger.info("Stage 2 raw output: %s", result_pil.size)

        # --- Deterministic post-processing (applied at Gemini resolution before upscaling) ---
        # Generate car mask once from the ORIGINAL small image — stable reference
        logger.info("Generating car mask for post-processing...")
        car_mask_small = _get_car_mask_rembg(pil_img_small)

        # Force wall/ceiling background to pure white — eliminates cyclorama shadow and
        # any grey arc Gemini leaves behind regardless of prompt instructions.
        result_pil = _force_wall_background_white(result_pil, car_mask_small)

        # Restore floor color to match the original (Gemini often shifts floor brightness)
        floor_clean_resized = floor_clean_pil.resize(result_pil.size, Image.Resampling.LANCZOS)
        result_pil = _restore_floor_from_original(floor_clean_resized, result_pil, car_mask_small)

        # -----------------------------------------------------------------------
        # Quality validation + automatic retry (Attempt 2)
        # Check attempt 1 result against the original:
        #   - Are reflections still visible on car panels?
        #   - Did the car color drift significantly?
        # If either check fails → build a targeted retry prompt and call Gemini
        # once more, then apply the same post-processing to the retry result.
        # The better result (fewer issues) is kept.
        # -----------------------------------------------------------------------
        reflections_remain, reflection_score = _check_reflections_remaining(
            pil_img_small, result_pil, car_mask_small
        )
        color_drifted, color_drift = _check_color_drift(
            pil_img_small, result_pil, car_mask_small
        )

        if reflections_remain or color_drifted:
            logger.info(
                "Attempt 1 quality check FAILED (reflections=%s score=%.1f%%, "
                "color_drift=%s drift=%.1f%%) — running attempt 2",
                reflections_remain, reflection_score * 100,
                color_drifted, color_drift * 100,
            )
            retry_prompt = _build_retry_prompt(
                reflections_remain, reflection_score, color_drifted, color_drift
            )
            try:
                # Send the attempt 1 result to Gemini with the targeted fix prompt
                retry_input_bytes = _pil_to_jpeg_bytes(result_pil)
                retry_response = _call_gemini_with_retry(
                    client, retry_prompt, retry_input_bytes, aspect,
                    f"{filename}[attempt2]"
                )
                retry_pil = _extract_image_from_response(retry_response)

                if retry_pil.size != result_pil.size:
                    retry_pil = retry_pil.resize(result_pil.size, Image.Resampling.LANCZOS)

                # Apply same post-processing to retry result
                retry_pil = _force_wall_background_white(retry_pil, car_mask_small)
                retry_pil = _restore_floor_from_original(floor_clean_resized, retry_pil, car_mask_small)

                # Validate retry result — keep whichever attempt is better
                retry_reflections, retry_ref_score = _check_reflections_remaining(
                    pil_img_small, retry_pil, car_mask_small
                )
                retry_color_drifted, retry_color_drift = _check_color_drift(
                    pil_img_small, retry_pil, car_mask_small
                )

                # Score: lower = better (sum of both issue scores)
                attempt1_score = reflection_score + color_drift
                attempt2_score = retry_ref_score  + retry_color_drift

                if attempt2_score < attempt1_score:
                    logger.info(
                        "Attempt 2 is better (score %.3f vs %.3f) — using attempt 2",
                        attempt2_score, attempt1_score,
                    )
                    result_pil = retry_pil
                else:
                    logger.info(
                        "Attempt 1 is better or equal (score %.3f vs %.3f) — keeping attempt 1",
                        attempt1_score, attempt2_score,
                    )
            except Exception as e:
                logger.warning("Attempt 2 failed (%s) — keeping attempt 1 result", e)
        else:
            logger.info("Attempt 1 quality check PASSED — no retry needed")

        logger.info("Stage 2 output received at %s — upscaling to %dx%d", result_pil.size, orig_w, orig_h)

    else:
        if background == "transparent":
            prompt = BACKGROUND_REMOVAL_TRANSPARENT_PROMPT
        else:
            prompt = BACKGROUND_REMOVAL_PROMPT

        response = _call_gemini_with_retry(client, prompt, img_bytes, aspect, filename)
        result_pil = _extract_image_from_response(response)
        logger.info("Gemini output received at %s — upscaling to %dx%d", result_pil.size, orig_w, orig_h)

    # Flip correction: Gemini sometimes mirrors the image horizontally.
    # Detect this by comparing the car mass position and column correlation
    # against the original, then flip back if needed.
    if _is_flipped(pil_img_small, result_pil):
        result_pil = ImageOps.mirror(result_pil)
        logger.info("Flip corrected: image was mirrored by Gemini, restored to original orientation")

    # Upscale Gemini output back to original full resolution (no pixel mixing or modifications)
    if result_pil.size != (orig_w, orig_h):
        result_pil = result_pil.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

    # Apply brightness boost if requested
    result_pil = _apply_brightness(result_pil, lighting_boost)

    # Convert to requested format at maximum quality.
    # NEF is Nikon's proprietary binary RAW format — cannot be written by any Python library.
    # TIFF is the correct lossless professional equivalent for processed RAW exports.
    out_buf = io.BytesIO()
    fmt = output_format.lower()
    if fmt == "png":
        result_pil.save(out_buf, format="PNG", compress_level=0)
    elif fmt in ("jpg", "jpeg"):
        result_pil.save(out_buf, format="JPEG", quality=100, subsampling=0)
    elif fmt == "webp":
        result_pil.save(out_buf, format="WEBP", quality=100, lossless=False)
    elif fmt in ("tif", "tiff", "nef"):
        # TIFF with LZW compression: lossless, smaller than uncompressed, widely supported
        # in Photoshop, Lightroom, and all professional editing software
        result_pil.save(out_buf, format="TIFF", compression="tiff_lzw")
    else:
        result_pil.save(out_buf, format="PNG", compress_level=0)

    return out_buf.getvalue()
