# What the Client Needs to Provide

This document lists everything the client (Richie) must provide or set up for the Car Image AI system to work fully.

---

## 1. Hugging Face (Required for AI processing)

**Purpose:** Download and run the RMBG-1.4 background removal model.

| Item | How to get it |
|------|---------------|
| **Hugging Face account** | Sign up at https://huggingface.co/join |
| **RMBG-1.4 access** | Go to https://huggingface.co/briaai/RMBG-1.4 and accept the license |
| **API token** | Create at https://huggingface.co/settings/tokens (read access is enough) |

**Where to use:** Set `HUGGINGFACE_HUB_TOKEN=your_token` in the backend environment (e.g. `.env` or server config).

---

## 2. Replicate (Required for enhance-preserve)

**Purpose:** AI inpainting for reflection removal. Two models available:

| Model | Env var | Cost | Quality |
|-------|---------|------|---------|
| **FLUX Fill Pro** (default) | `REPLICATE_INPAINT_MODEL=flux` | ~$0.03/image | Best |
| **SD Inpainting** (open-source) | `REPLICATE_INPAINT_MODEL=sd-inpainting` | ~$0.004/image | Good |

| Item | How to get it |
|------|---------------|
| **Replicate account** | Sign up at https://replicate.com |
| **API token** | Create at https://replicate.com/account/api-tokens |

**Where to use:** Set `REPLICATE_API_TOKEN=your_token` in `backend/.env`.

Optional: Add `REPLICATE_INPAINT_MODEL=sd-inpainting` to use the cheaper Stable Diffusion inpainting model instead of FLUX.

Without the token, enhance-preserve falls back to local OpenCV inpainting (lower quality).

---

## 3. Google Drive (Optional – for “Import from Google Drive”)

**Purpose:** Allow users to pick and import images directly from Google Drive.

### Setup steps (client or developer does these)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select a project
3. Enable **Google Picker API** and **Google Drive API**
   - APIs & Services → Library → search for each → Enable
4. Create credentials:
   - **OAuth 2.0 Client ID** (Web application)
     - Authorized JavaScript origins: `http://localhost:3000`, `https://your-domain.com`
     - Authorized redirect URIs: same as above
   - **API Key**
     - Create API key and restrict to Picker and Drive APIs
5. Get **Project number** (App ID)
   - IAM & Admin → Settings → Project number

### Values to provide to the developer

| Variable | Description |
|----------|-------------|
| `NEXT_PUBLIC_GOOGLE_CLIENT_ID` | OAuth 2.0 Client ID (e.g. `xxx.apps.googleusercontent.com`) |
| `NEXT_PUBLIC_GOOGLE_API_KEY` | API key for Picker/Drive APIs |
| `NEXT_PUBLIC_GOOGLE_APP_ID` | Project number (App ID) |

**Where to use:** In `frontend/.env.local` or deployment environment variables.

Without these, the “Import from Google Drive” button stays disabled.

---

## 4. NEF Support (Already implemented)

No extra setup from the client. NEF (Nikon RAW) is supported out of the box.

---

## 5. Hosting (Optional – if using your own server)

| Item | Notes |
|------|-------|
| **VPS** | Already have Hostinger (4GB RAM, 50GB) |
| **Domain** | Optional – can use IP for testing |
| **HTTPS** | Needed for Google OAuth in production |

---

## Summary checklist

| Feature | Client action needed |
|---------|----------------------|
| **Background removal (core)** | Hugging Face token |
| **Reflection removal (enhance-preserve)** | Replicate token (FLUX.1-Fill-dev — primary model) |
| **Google Drive import** | Google Cloud project + Client ID, API Key, App ID |
| **NEF files** | Nothing |
| **VPS hosting** | Already set up |
