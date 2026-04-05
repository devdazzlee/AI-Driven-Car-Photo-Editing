"""
Gemini API service for car image processing.

Uses Gemini API (gemini-3.1-flash-image-preview) for all image processing.
Single API call: remove reflections, clean floor, maintain car color, keep walls/floor intact.
"""

import base64
import io
import logging
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
    "Edit this car dealership photo with these exact instructions:\n\n"

    "STEP A — SAMPLE AND LOCK THE CAR'S TRUE PAINT COLOR (DO THIS FIRST, BEFORE ANY EDITING):\n"
    "Before making any changes, look at the car and find the DARKEST, LEAST-AFFECTED area of "
    "each major painted panel — the area least hit by studio light. That darkest area is the "
    "car's TRUE paint color. Memorize it. Every reflection you remove must be replaced with "
    "EXACTLY this true color — not a lighter version, not a different shade. The true color "
    "is your anchor. You must not drift from it.\n"
    "   - BLACK/VERY DARK car: true color = deepest black area on any door or fender bottom edge\n"
    "   - DARK BLUE/NAVY car: true color = deepest blue area, NOT the lighter blue-grey zones\n"
    "   - DARK GREEN car: true color = deepest green area on the panel bottom\n"
    "   - WHITE car: true color = the smooth even white away from any bright blobs\n"
    "   - GREY/SILVER car: true color = the mid-tone metallic grey in the least-lit zone\n"
    "   - ANY OTHER COLOR: true color = the richest, most saturated area of that color on the panel\n"
    "CRITICAL: Studio lights always make paint look LIGHTER than its true color. The true color "
    "is always the DARKER, MORE SATURATED version you see in shadow areas or panel bottom edges. "
    "Never use a light/bright zone as your reference — that zone is contaminated by studio light.\n\n"

    "STEP B — MEMORIZE THE FLOOR IDENTITY (FOR VERIFICATION — NOT FOR REGENERATION):\n"
    "Immediately after locking the car color, look at the floor in the input image and "
    "memorize these specific properties — you will use them to verify your output:\n"
    "   1. TILE SIZE: Large tiles (30cm+)? Medium? Small? Measure relative to the car wheel.\n"
    "   2. TILE MATERIAL: Stone/slate with texture? Smooth concrete? Ceramic? Rough or smooth?\n"
    "   3. TILE COLOR: Dominant color — dark grey-brown? Light grey? Beige? Be specific.\n"
    "   4. TILE DARKNESS: How dark overall on a scale 1-10?\n"
    "   5. GROUT: Wide grout lines? Narrow? Same color as tile or contrasting?\n"
    "These five values = the FLOOR IDENTITY. Your output floor must match ALL FIVE.\n"
    "This is a READ-ONLY record for verification. It does NOT give you permission to "
    "recolor, regenerate, or reimagine the floor. The floor must stay exactly as it is.\n\n"

    "1. COMPOSITION AND FRAMING — THE CAR MUST NOT MOVE (ABSOLUTE RULE):\n"
    "   This is not a creative photo shoot. You are editing the background only. "
    "The car's position, angle, and framing are LOCKED and must not change under any circumstances.\n"
    "   - Do NOT change the camera angle, perspective, zoom level, or viewing angle.\n"
    "   - The car's viewing angle in the output must be IDENTICAL to the input. "
    "If the input shows a front-left three-quarter view, the output must show the same "
    "front-left three-quarter view — not more front-on, not more side-on. Same exact angle.\n"
    "   - Do NOT rotate or reposition the car. Do NOT zoom in or out. "
    "Do NOT reframe or recompose the shot.\n"
    "   - Do NOT flip or mirror the image horizontally or vertically.\n"
    "   - CRITICAL: Even when removing the garage door, studio equipment, or cleaning the "
    "background — the car must NOT move. When background elements are removed and replaced "
    "with white, the car stays in exactly the same pixel position it was in the input. "
    "Do not adjust the car's position to 'fill' the cleaned background area.\n"
    "   - The license plate must appear at the exact same position and size in the frame.\n"
    "   - Never cut off bumpers, mirrors, roof, hood, trunk or any other part of the car.\n"
    "   - Every part visible in the original must remain visible and in the same position.\n"
    "   - Return the image at exactly the same dimensions as the input.\n\n"

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
    "   Look at the top half of your output. Ask yourself: do I see ANY grey shape, curved arc, "
    "dome, gradient, or shadow in the background? If YES — paint it white. Do not proceed to "
    "step 3 or 4 until the answer is NO. The background is only complete when it is pure flat "
    "white with absolutely no grey anywhere.\n\n"
    "   RULE A — ALSO REMOVE ALL OF THESE (replace every item below with pure white #FFFFFF):\n"
    "   GARAGE DOOR: If you see a garage door in the background — a large rectangular door "
    "with horizontal panel sections and metal hinges, typically on the right or left side of "
    "the background — REPLACE IT ENTIRELY WITH FLAT WHITE. The entire garage door and its "
    "frame must disappear and become pure white. This is not optional.\n"
    "   ALSO REMOVE: Studio lights, light fixtures, ceiling rigs, overhead cables, "
    "door frames, hinges, handles, and any other equipment mounted on walls or ceiling.\n"
    "   CRITICAL EXCEPTION — DO NOT REMOVE CAR PARTS: Never remove any element physically "
    "attached to the car. This includes WITHOUT EXCEPTION:\n"
    "     * Radio antenna, shark fin antenna, mast antenna on the roof\n"
    "     * Windshield wipers, side mirror mounts, roof rails\n"
    "     * STEP BARS and RUNNING BOARDS — the metal bars/steps running along the bottom "
    "side of vans, trucks, and SUVs that help passengers step up into the vehicle. "
    "These are car components, NOT studio equipment. Do NOT remove them.\n"
    "     * Ladder racks, roof cargo racks mounted ON THE VEHICLE (not on the studio wall)\n"
    "     * Any trim, panel, or component touching or part of the car body\n"
    "   Only remove objects that are mounted on the WALL or CEILING of the studio.\n"
    "   RULE B — THE RESULT: The entire wall and ceiling area = perfectly flat pure white "
    "#FFFFFF. No variation. No grey. No gradients. No curved shapes. No shadows anywhere. "
    "Every off-white, grey, or tinted pixel above the floor level (that is not the car) "
    "must be #FFFFFF. This is non-negotiable.\n"
    "   The ONLY exception: the wall-floor junction line may remain visible and natural.\n\n"

    "3. BACKGROUND — FLOOR:\n\n"
    "   *** IMPORTANT — THE FLOOR IN THIS IMAGE HAS ALREADY BEEN PRE-CLEANED ***:\n"
    "   This image has been through a floor-cleaning pre-process. The floor you see in this "
    "input image is ALREADY CLEAN. The floor color you see IS CORRECT — it matches the "
    "original tile color.\n"
    "   YOUR JOB FOR THE FLOOR: PRESERVE IT EXACTLY. Do NOT re-clean it, do NOT 'improve' it, "
    "do NOT make it darker, do NOT make it lighter. The floor in your output must look "
    "IDENTICAL to the floor in this input — same color, same tile shade, same darkness.\n"
    "   WARNING: Do not default to a darker or more 'professional' floor color. The shade "
    "of grey (or other color) you see in this input IS the correct shade. Preserve it.\n\n"

    "   THE PRIMARY RULE FOR FLOOR IN THIS STAGE: Copy it exactly. "
    "The floor color, tile shade, tile darkness, tile material, grout — ALL must be PIXEL-IDENTICAL "
    "to what you see in this input image. Do not make it darker. Do not make it lighter. "
    "Do not 'improve' it. Just copy it faithfully into the output.\n\n"
    "   FORBIDDEN FLOOR CHANGES — any of these = edit has FAILED:\n"
    "   - Output floor is darker than input floor → FORBIDDEN\n"
    "   - Output floor is lighter than input floor → FORBIDDEN\n"
    "   - Output tiles are a different size → FORBIDDEN\n"
    "   - Output tile material changed → FORBIDDEN\n"
    "   - Output grout is different → FORBIDDEN\n\n"
    "   IF you spot any remaining dirty mark the pre-clean missed:\n"
    "   Remove it by filling with the IMMEDIATELY ADJACENT tile color. "
    "Do not use a darker or lighter fill — match the surrounding tiles exactly. "
    "Check under the car, between tires, in front of bumper — all zones.\n\n"
    "   FLOOR VERIFICATION:\n"
    "   [ ] Floor tile color same darkness/lightness as this input image?\n"
    "   [ ] Floor tile material/size unchanged?\n"
    "   [ ] Zero dirty spots remaining?\n"
    "   If floor color changed (darker or lighter) → REDO floor section.\n\n"

    "4. REFLECTIONS — REMOVE STUDIO LIGHT HOTSPOTS WITHOUT CHANGING THE CAR'S COLOR:\n"
    "   You sampled and locked the true paint color above. Now use it.\n\n"

    "   THE ABSOLUTE RULE — COLOR MUST NOT CHANGE:\n"
    "   Reflection removal means ONLY removing the overexposed brightness added by studio lights. "
    "It does NOT mean repainting the car. After reflection removal:\n"
    "   → The car's base paint color must be IDENTICAL to the original\n"
    "   → A dark navy car must still be dark navy — not lighter blue, not grey-blue\n"
    "   → A black car must still be black — not dark grey, not charcoal\n"
    "   → A white car must still be the same white — not brighter, not duller\n"
    "   → ANY color shift in the paint after processing = you changed the car color = WRONG\n"
    "   Fill every removed hotspot with the TRUE paint color you locked above — the darkest, "
    "most saturated version from the unaffected panel edges and bottom areas.\n\n"

    "   HOW TO REMOVE A REFLECTION WITHOUT CHANGING COLOR:\n"
    "   Step 1: Find the hotspot (area significantly brighter than surrounding same-panel paint)\n"
    "   Step 2: Sample the TRUE paint color from the darkest unaffected corner of that same panel\n"
    "   Step 3: Fill the hotspot area with that exact true color, blending naturally at edges\n"
    "   Step 4: Verify — does the treated area now match the surrounding paint exactly? "
    "Same hue? Same saturation? Same darkness? If not → fix it.\n"
    "   NEVER sample from another bright or lit zone. Only sample from dark, shadow, or "
    "bottom-edge areas of the panel where the true paint color is undistorted by light.\n\n"

    "   WHAT IS A STUDIO REFLECTION (identify these and remove them):\n"
    "   A reflection is a zone that is LOCALLY BRIGHTER than the true paint color.\n"
    "   It looks like a bright band, blob, or wash of light sitting ON TOP of the paint.\n"
    "   For every car color, reflections look like this on that color:\n"
    "     - BLACK car: grey, silver, or white patches/streaks on black panels\n"
    "     - DARK BLUE/NAVY car: light blue, grey-blue, or white-blue patches on dark blue panels\n"
    "     - DARK GREEN car: light green or grey-green patches on dark green panels\n"
    "     - WHITE car: extra-bright harsh white blobs brighter than the smooth surrounding white\n"
    "     - GREY car: near-white or washed-out zones on mid-grey panels\n"
    "     - ANY COLOR: wherever paint looks faded, washed-out, or lighter than it should be\n\n"
    "   CRITICAL — METALLIC AND SILVER PAINT (READ CAREFULLY):\n"
    "   Silver, metallic silver, and metallic paint of any color naturally have VARYING "
    "BRIGHTNESS across the same panel. This variation is caused by metallic flakes catching "
    "light at different angles — it is the natural appearance of metallic paint, NOT a "
    "studio reflection. DO NOT touch this natural metallic variation.\n"
    "   For a SILVER/METALLIC car:\n"
    "     - The same door panel will look lighter in some zones and darker in others — KEEP THIS\n"
    "     - The hood may have areas that look slightly lighter or more silver — KEEP THIS\n"
    "     - DO NOT flatten or homogenize the metallic panels to a uniform grey\n"
    "     - DO NOT make any silver panel look like a solid flat grey — it must retain metallic depth\n"
    "   Only remove EXTREME overexposed white/near-white blobs that are clearly too bright to be "
    "any metallic paint — zones that look blown-out and washed-out rather than metallic.\n"
    "   After removing a reflection from metallic paint: the area must still look metallic — "
    "it must still have the same silvery sheen and depth as the rest of that panel, not flat grey.\n\n"

    "   SCAN EVERY PANEL — REMOVE HOTSPOTS, RESTORE TRUE COLOR:\n"
    "   → ROOF: find true roof color at panel edges → remove all brighter zones → fill with true color\n"
    "   → HOOD: find true hood color at base → remove all brighter zones → fill with true color\n"
    "   → FRONT FENDER: remove bright patches → restore true paint color\n"
    "   → FRONT DOOR: remove every bright streak/blob → restore true paint color\n"
    "   → REAR DOOR(S): same treatment\n"
    "   → REAR QUARTER PANEL: remove bright zones especially upper curve\n"
    "   → TAILGATE/BED COVER/REAR PANEL: remove all bright zones\n"
    "   → ALL PILLARS (A, B, C, D): remove hotspots\n"
    "   → FRONT AND REAR BUMPERS: remove hotspots\n\n"

    "   GLASS SURFACES — STEP-BY-STEP REFLECTION REMOVAL (EVERY WINDOW MUST BE TREATED):\n"
    "   Glass is the most reflective surface on a car. Studio lights create broad white/grey washes "
    "across glass that make windows look opaque and blown-out. Every window must be fixed.\n\n"
    "   WHAT A GLASS REFLECTION LOOKS LIKE:\n"
    "     - A bright white, light grey, or washed-out zone on a window pane\n"
    "     - True clean glass (no reflection): deep dark grey/charcoal — you are looking at the dark "
    "car interior through the glass, and it should look like a dark mirror, NOT a frosted pane\n"
    "     - Any part of a window lighter than dark charcoal grey = studio light reflection = remove it\n\n"
    "   HOW TO REMOVE GLASS REFLECTION (USE THIS METHOD FOR EVERY WINDOW):\n"
    "   Step G1: Find the DARKEST unaffected corner of that window pane — the spot where no studio "
    "light hit directly. That corner = the TRUE glass tone.\n"
    "   Step G2: The true glass tone will be dark grey, near-black, or deep charcoal.\n"
    "   Step G3: Fill EVERY area of that same window that is lighter than this dark reference "
    "with the same dark tone, blending naturally at edges. The entire glass pane must match "
    "the tone of its own darkest corner.\n"
    "   Step G4: Verify — the window frame/rubber seal border is still visible, and the glass "
    "pane itself is uniformly dark with NO bright blobs, NO white patches, NO light grey wash.\n"
    "   NEVER leave any window lighter than dark grey — anything lighter = reflection still present.\n\n"
    "   TREAT EACH WINDOW INDIVIDUALLY:\n"
    "   → WINDSHIELD: sample darkest corner → fill every bright/white zone → uniform dark grey result.\n"
    "     Keep the windshield wiper shape (car part — do NOT remove). "
    "Keep the rear-view mirror base at top (car part — do NOT remove).\n"
    "   → FRONT DOOR WINDOWS (driver + passenger): apply Step G1–G4. Both windows must match each "
    "other in darkness. Any diagonal or vertical bright streak is a studio light band — remove it.\n"
    "   → REAR DOOR WINDOWS: same treatment. Must match front door windows in darkness.\n"
    "   → REAR QUARTER GLASS (small triangular/trapezoidal pane behind rear door, beside C-pillar):\n"
    "     *** THIS IS A HIGH-RISK PROBLEM AREA — DO NOT SKIP IT ***\n"
    "     This small pane persistently retains grey/white studio glow if not explicitly treated.\n"
    "     Step G1: find its darkest corner. Step G3: fill everything lighter with that dark tone.\n"
    "     After fixing: rear quarter glass must be the SAME DARKNESS as the door window beside it.\n"
    "     If it still looks 'slightly grey' — that IS a remaining reflection. Fix it.\n"
    "   → REAR WINDOW (entire back windshield — large rear glass panel):\n"
    "     *** THIS IS THE OTHER HIGH-RISK PROBLEM AREA — DO NOT SKIP IT ***\n"
    "     The rear window is the largest glass surface and collects the broadest studio reflection — "
    "a wide light-grey wash that covers most of the pane, not just a small blob.\n"
    "     Step G1: find the absolute darkest zone in the rear window (usually a low corner or edge).\n"
    "     Step G3: fill the ENTIRE rear window — top, centre, AND bottom — to that dark tone.\n"
    "     Pay special attention to:\n"
    "       * The TOP of the rear window where it meets the roofline — usually bright white here\n"
    "       * The CENTRE of the rear window — usually has broad grey wash from overhead studio light\n"
    "     If the rear window has a defroster grid (thin horizontal lines across the glass): "
    "KEEP those lines — they are car components. But the glass BETWEEN those lines must be DARK.\n"
    "     After fixing: the entire rear window must look like a dark mirror — uniformly dark, "
    "no grey wash anywhere, no bright zones anywhere. Same darkness as door windows.\n\n"
    "   GLASS FINAL CHECK (ALL MUST PASS):\n"
    "   [ ] Windshield: uniformly dark, no bright patches\n"
    "   [ ] Front door windows: dark, matching each other\n"
    "   [ ] Rear door windows: dark, same tone as front\n"
    "   [ ] Rear quarter glass: DARK — as dark as the door window beside it (if grey/bright → FIX)\n"
    "   [ ] Rear window: DARK throughout — no grey wash, no bright top zone (if not dark → FIX)\n"
    "   Any remaining light/grey glass = unremedved studio reflection = edit is incomplete.\n\n"

    "   SIDE MIRROR:\n"
    "   Mirror HOUSING (outer plastic/painted shell) = car body color = keep exactly unchanged.\n"
    "   Mirror FACE (reflective glass disc) = blown-out white from studio light = IS a reflection. "
    "Replace with natural dark grey/silver as if reflecting a neutral environment.\n\n"

    "   SECTION 4 FINAL CHECK — DO NOT RETURN UNTIL ALL PASS:\n"
    "   [ ] Car paint color — is it the SAME color as original? Same hue, same saturation, same darkness?\n"
    "   [ ] All body panels (roof, hood, doors, fenders, quarters, tailgate, bumpers) — free of bright "
    "streaks, blobs, washed-out zones? Every hotspot filled with the locked true paint color?\n"
    "   [ ] Windshield — dark, no bright patches?\n"
    "   [ ] Front door windows — dark, matching each other?\n"
    "   [ ] Rear door windows — dark?\n"
    "   [ ] REAR QUARTER GLASS — dark grey, same as door window beside it? (This is the most commonly "
    "missed glass surface — check it specifically)\n"
    "   [ ] REAR WINDOW — dark throughout, no grey wash at top or centre? (This is the most commonly "
    "missed large glass surface — check it specifically)\n"
    "   [ ] Mirror face — dark grey, not white?\n"
    "   If any check fails → fix that specific item before returning. Do not return with a single "
    "remaining reflection or bright glass surface — each is visible and constitutes a failed edit.\n\n"

    "   DESIGN FEATURES — NEVER TOUCH:\n"
    "   → Chrome trim strips on window frames, roof rails, door sills — keep exactly\n"
    "   → Chrome door handle levers — keep exactly\n"
    "   → Chrome/silver grille bars — keep exactly\n"
    "   → Razor-thin specular line exactly on a sharp pressed body crease — keep\n"
    "   → Mirror HOUSING color — keep exactly\n\n"

    "5. CAR COLOR — ABSOLUTE PRESERVATION (THIS OVERRIDES EVERYTHING ELSE):\n"
    "   The car's paint color in the OUTPUT must be identical to the INPUT — same hue, same "
    "saturation, same darkness, same finish character. This is non-negotiable and overrides "
    "any other instruction. Specifically:\n"
    "   - A dark navy blue car must remain dark navy blue — NOT lighter blue, NOT grey-blue\n"
    "   - A black car must remain black — NOT dark grey, NOT charcoal\n"
    "   - A dark green car must remain dark green — NOT lighter green\n"
    "   - A red car must remain the same red — NOT orange-red, NOT darker red\n"
    "   - A white car must remain the same white — NOT brighter, NOT cream\n"
    "   - A silver car must remain the same silver grey — NOT lighter, NOT darker\n"
    "   If you compare the output car paint to the input car paint and they look different "
    "in color — the edit has FAILED on color preservation. The reflection removal in Section 4 "
    "must be done in a way that preserves color exactly. If removing a reflection is causing "
    "color change, you are sampling the wrong reference color — resample from the darkest "
    "unaffected area of the panel and try again.\n\n"

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
    "   - Do NOT change the color of ANY car part or component. "
    "Headlights, tail lights, headlight surrounds, grille, grille surround, bumpers, body panels, "
    "hood, roof, doors, door handles, side mirrors, pillar trim, step bars, running boards, "
    "and every other component must remain EXACTLY the same color as in the original photo. "
    "No color changes to any car part are acceptable under any circumstances.\n"
    "   - Do NOT add, create, invent or hallucinate any car parts, badges, decorative elements, "
    "chrome surrounds, lighting elements, or any other components that are NOT visible in the "
    "original image. You must never add anything new to the car — only clean the background.\n"
    "   - Do NOT repaint, recolor, alter or modify any part of the car body, trim, or components. "
    "The car must look physically identical to the original — same parts, same colors, same finish. "
    "If a part is black in the original it must stay black. "
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

    "13. CAR ANTENNA — DO NOT REMOVE (ABSOLUTE RULE):\n"
    "   The car has a radio antenna / mast antenna mounted on the roof or body. It appears as a "
    "thin vertical or slightly angled line/rod protruding from the top of the car. "
    "This is a car component — NOT a studio cable or wire. "
    "You MUST keep the antenna fully intact and visible in the output, exactly as it appears in "
    "the original. Do not remove it, erase it, shorten it, or paint over it with the background "
    "color. Removing the antenna is strictly forbidden.\n\n"

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
    "Remove the background from this car photo. "
    "Replace the background with a clean solid white background. "
    "Keep the car exactly as it is - preserve all details, colors, and reflections. "
    "Return only the edited image, no text."
)

BACKGROUND_REMOVAL_TRANSPARENT_PROMPT = (
    "Remove the background from this car photo. "
    "Make the background fully transparent. "
    "Keep the car exactly as it is - preserve all details, colors, and reflections. "
    "Return only the edited image with transparent background, no text."
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
                    image_config=types.ImageConfig(aspect_ratio=aspect, image_size="1K"),
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
    """Detect horizontal flip using car mass position and column correlation."""
    size = (128, 128)
    orig_gray = np.array(original.resize(size).convert("L"), dtype=np.float64)
    res_gray = np.array(result.resize(size).convert("L"), dtype=np.float64)

    # Method 1: Car mass center-of-gravity
    orig_cx = _find_car_center_x(orig_gray)
    res_cx = _find_car_center_x(res_gray)
    mass_flipped = False
    cx_diff = abs(orig_cx - res_cx)
    if cx_diff > 0.1:
        mirrored_cx = 1.0 - res_cx
        if abs(orig_cx - mirrored_cx) < abs(orig_cx - res_cx):
            mass_flipped = True

    # Method 2: Column profile correlation
    orig_profile = orig_gray.mean(axis=0)
    res_profile = res_gray.mean(axis=0)
    res_flipped = res_profile[::-1]
    normal_corr = np.corrcoef(orig_profile, res_profile)[0, 1]
    flipped_corr = np.corrcoef(orig_profile, res_flipped)[0, 1]
    corr_flipped = flipped_corr > normal_corr and abs(flipped_corr - normal_corr) > 0.02

    is_flip = mass_flipped or corr_flipped
    if is_flip:
        logger.warning("Flip detected (mass=%s, corr=%s) — correcting", mass_flipped, corr_flipped)
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
                          mask_np: np.ndarray | None = None) -> Image.Image:
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


def _get_car_mask_rembg(pil_img: Image.Image) -> np.ndarray:
    """Get boolean mask of the car using rembg."""
    try:
        from rembg import remove
        mask_pil = remove(pil_img, only_mask=True)
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
    h, w = car_mask_np.shape
    background_mask = ~car_mask_np
    floor_row_start = int(h * 0.60)
    floor_row_mask = np.zeros((h, w), dtype=bool)
    floor_row_mask[floor_row_start:, :] = True
    floor_mask = background_mask & floor_row_mask
    if not floor_mask.any():
        return (200, 200, 200)
    img_np = np.array(pil_img)
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

    h, w = car_mask_np.shape
    orig_np = np.array(original).astype(np.float32)
    proc_np = np.array(processed).astype(np.float32)

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

    Scope: ONLY non-car pixels in the upper portion of the frame (wall/ceiling zone).
    The floor and all car pixels are never touched by this function.

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

    # Wall/ceiling zone: top 58% of frame rows
    wall_row_end = int(ph * 0.58)
    wall_row_mask = np.zeros((ph, pw), dtype=bool)
    wall_row_mask[:wall_row_end, :] = True

    # Apply only to background (non-car) pixels in wall zone
    wall_bg_mask = (~car_mask) & wall_row_mask

    result_np = proc_np.copy()
    result_np[wall_bg_mask] = [255, 255, 255]

    logger.info(
        "Wall background: forced %d pixels to pure white (wall/ceiling zone, top 58%%)",
        int(wall_bg_mask.sum()),
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
        logger.info("Stage 2 output received at %s — upscaling to %dx%d", result_pil.size, orig_w, orig_h)

    else:
        if background == "transparent":
            prompt = BACKGROUND_REMOVAL_TRANSPARENT_PROMPT
        else:
            prompt = BACKGROUND_REMOVAL_PROMPT

        response = _call_gemini_with_retry(client, prompt, img_bytes, aspect, filename)
        result_pil = _extract_image_from_response(response)
        logger.info("Gemini output received at %s — upscaling to %dx%d", result_pil.size, orig_w, orig_h)

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
