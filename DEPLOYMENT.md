# Deploying Drugenix (free, public link)

Two pieces get deployed separately: the **backend** (FastAPI, does the actual
work) goes to Render; the **frontend** (React/Vite, the UI) goes to Vercel.
Both have free tiers and both stay free at your usage level. Total cost: $0,
as long as you keep the Gemini API key on its free tier too (see the note
on rate limits at the bottom).

You already have a GitHub repo (`CodeCraftsmanRaj/Drugenix`, branch `rasika`)
— both platforms deploy straight from it, so push your latest changes first:

```
git add -A
git commit -m "Production hardening + deployment config"
git push origin rasika
```

## 1. Backend on Render

1. Go to https://render.com and sign up / log in (GitHub login is easiest).
2. Click **New +** → **Blueprint**.
3. Connect your GitHub account if prompted, then select the `Drugenix` repo.
4. Render will detect `render.yaml` in the repo root and pre-fill everything
   (service name `drugenix-backend`, build/start commands, free plan).
5. Click **Apply**. It'll ask you to fill in one value it deliberately left
   blank: **GEMINI_API_KEY** — paste your key from https://aistudio.google.com/apikey.
6. Deploy takes a few minutes the first time. When it's done, Render gives
   you a URL like `https://drugenix-backend.onrender.com` — **copy this**,
   you need it for step 2.
7. Sanity check: open `https://drugenix-backend.onrender.com/api/health` in
   a browser — you should get back JSON, not an error page.

**Free tier behavior to expect:** Render's free web services sleep after
15 minutes of no traffic. The first request after sleeping takes ~30-50
seconds to wake back up (later requests are normal speed). If you're doing
a live demo, open the URL and let it wake up a minute or two before you
start, so nobody's staring at a spinner.

## 2. Frontend on Vercel

1. Go to https://vercel.com and sign up / log in (GitHub login is easiest).
2. Click **Add New** → **Project**, select the `Drugenix` repo.
3. Vercel will ask for the project root — set it to `frontend` (not the
   repo root, since that's where `package.json` actually lives).
4. Framework preset should auto-detect as **Vite**. Leave build command
   (`npm run build`) and output directory (`dist`) as default.
5. Before deploying, add one environment variable:
   - **Name:** `VITE_API_BASE`
   - **Value:** the Render URL from step 1 (e.g. `https://drugenix-backend.onrender.com`)
     — no trailing slash.
6. Click **Deploy**. You'll get a public URL like `https://drugenix.vercel.app`
   — that's the link you share with mentors/industry experts.

Any time you `git push` to the `rasika` branch afterward, both Render and
Vercel automatically redeploy — you don't need to repeat these steps.

## 3. Before you demo

- Open the Vercel link yourself first and run one query, to both wake up
  the sleeping Render backend and confirm the Gemini key is working (check
  for the red data-quality banner — if you see it, read what it says).
- Free-tier Gemini is capped at 20 requests per window. A single symptom
  query can use several of those. If you're demoing to a room, don't fire
  off many queries back-to-back right before — space them out, or set
  "Conditions to explore" to 1 in the symptom form to use fewer calls per
  query. If you decide the quota risk isn't worth it later, enabling
  billing on the same Gemini key removes this limit entirely (this was
  discussed earlier — no code changes needed, just a Google-side setting).
