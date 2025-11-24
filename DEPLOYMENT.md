# Streamlit Cloud Deployment Guide

Complete guide to deploy X-Ray Transparency Lab on Streamlit Cloud.

**Live Example**: https://mu-niu13-x-ray-transparency-lab-app-y74grj.streamlit.app/

---

## Prerequisites

- [ ] GitHub account
- [ ] Hugging Face account
- [ ] Trained model files locally (models/ and embeddings/)
- [ ] Git installed

---

## Step 1: Upload Model to Hugging Face

### 1.1 Create Hugging Face Account
1. Go to https://huggingface.co/join
2. Complete registration
3. Verify your email

### 1.2 Get Access Token
1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: `xray-model-upload`
4. Role: Select "Write"
5. Copy the token (starts with `hf_...`)

### 1.3 Login Locally
```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Login with your token
huggingface-cli login
# Paste token when prompted
```

### 1.4 Upload Your Model
```bash
# Make sure you're in the project directory
cd X-Ray-Transparency-Lab

# Check you have these files:
ls models/pneumonia_classifier.pth
ls embeddings/embeddings.npy
ls embeddings/labels.npy
ls embeddings/paths.pkl
ls embeddings/similarity_index.faiss

# Upload to Hugging Face
python upload_to_huggingface.py

# Wait for upload (5-10 minutes for ~140MB)
```

### 1.5 Verify Upload
After upload completes, you'll see:
```
✅ Model available at: https://huggingface.co/YOUR_USERNAME/xray-pneumonia-model
```

Visit that URL to confirm your files are uploaded.

**Your Repo ID**: `YOUR_USERNAME/xray-pneumonia-model`
(Copy this - you'll need it later!)

---

## Step 2: Prepare GitHub Repository

### 2.1 Clean Up Repository

Delete files no longer needed:
```bash
# Optional: Remove training/setup scripts not needed for deployment
rm download_trained_model.py
rm download_with_gdown.py
rm diagnose_setup.py
rm .env.example
```

### 2.2 Update .gitignore
Make sure these are in `.gitignore`:
```
models/
embeddings/
data/
*.zip
.env
.streamlit/secrets.toml
```

### 2.3 Commit and Push
```bash
git add .
git commit -m "Deploy with Hugging Face"
git push origin main
```

---

## Step 3: Deploy to Streamlit Cloud

### 3.1 Access Streamlit Cloud
1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Authorize Streamlit Cloud to access your repositories

### 3.2 Create New App
1. Click "New app" button
2. Fill in:
   - **Repository**: `Mu-niu13/X-Ray-Transparency-Lab` (your repo)
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL** (optional): Choose custom URL or use default

### 3.3 Configure Advanced Settings

Click "Advanced settings":

**Python version**: `3.10`

**Secrets**: Click "Secrets" and add:
```toml
HUGGINGFACE_REPO = "YOUR_USERNAME/xray-pneumonia-model"
```

**Important**: 
- Replace `YOUR_USERNAME/xray-pneumonia-model` with YOUR actual repo ID from Step 1.5
- No quotes around the value
- Example: `HUGGINGFACE_REPO = "Mu-niu13/xray-pneumonia-model"`

### 3.4 Deploy!
1. Click "Deploy!"
2. Watch the build logs
3. Wait 5-10 minutes for first deployment

### 3.5 Monitor Deployment

You'll see these stages:
1. ✅ Building container
2. ✅ Installing dependencies (from requirements.txt)
3. ✅ Starting app
4. ⬇️ Downloading model files (first run only, ~2-3 min)
5. ✅ App is live!

---

## Step 4: Test Your Deployment

### 4.1 Access Your App
Once deployed, you'll get a URL like:
```
https://your-username-x-ray-transparency-lab-app-xxxxx.streamlit.app/
```

### 4.2 First Run Test
1. App will show "Downloading model files..." (first time only)
2. Wait 2-3 minutes for download to complete
3. You'll see "Model files ready!"

### 4.3 Functional Test
1. Click "Use sample image instead" checkbox
2. Select a sample from dropdown
3. Click "Run AI Analysis"
4. Verify:
   - ✅ Prediction appears
   - ✅ Confidence score shows
   - ✅ Grad-CAM heatmap displays
   - ✅ Occlusion map displays
   - ✅ Similar cases show
   - ✅ Report generates

### 4.4 Upload Test
1. Upload your own chest X-ray image
2. Run analysis
3. Verify all features work

---

## Step 5: Optimization (Optional)

### 5.1 Custom Domain
Free tier includes `.streamlit.app` domain.

For custom domain:
1. Upgrade to Streamlit Cloud Pro
2. Follow their custom domain guide

### 5.2 Performance Monitoring
- Check app analytics in Streamlit dashboard
- Monitor resource usage
- View error logs if issues occur

### 5.3 Caching Strategy
Already implemented in `app.py`:
```python
@st.cache_resource
def download_model_files():
    # Downloads only once, then cached
```

---

## Troubleshooting

### Error: "HUGGINGFACE_REPO not found"

**Solution**:
1. Go to Streamlit Cloud dashboard
2. Click your app → Settings → Secrets
3. Add:
   ```
   HUGGINGFACE_REPO = "username/repo-name"
   ```
4. Click "Save"
5. App will automatically restart

### Error: "Module not found: huggingface_hub"

**Solution**:
Check `requirements.txt` includes:
```
huggingface-hub>=0.19.0
```

### Error: Model download fails

**Possible causes**:
1. **Repository is private**: Make your HF repo public
2. **Wrong repo ID**: Verify format is `username/repo-name`
3. **Files missing**: Check all files uploaded to HF

**Test manually**:
```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="YOUR_REPO_ID",
    filename="models/pneumonia_classifier.pth"
)
```

### Error: "Out of memory"

**Free tier limitations**:
- 1GB RAM
- 1 CPU core
- 1 concurrent user

**Solutions**:
1. Reduce batch processing
2. Clear session state more frequently
3. Upgrade to Pro tier ($20/month) for 3GB RAM

### App is slow

**First run**: 
- Downloads 140MB of model files (2-3 min)
- Files are cached afterward
- Subsequent starts are fast (<30 seconds)

**Ongoing slowness**:
- Check if processing large images
- Reduce occlusion patch stride in `explanations.py`
- Use lower resolution images

### Build fails

**Check logs for**:
- Python version mismatch (use 3.10)
- Missing dependencies in `requirements.txt`
- Syntax errors in code

**Fix**:
1. Test locally first: `streamlit run app.py`
2. Check all imports work
3. Push fix to GitHub
4. Streamlit Cloud auto-redeploys

---

## Maintenance

### Update Model

To deploy a new model version:

1. **Upload to Hugging Face**:
   ```bash
   python upload_to_huggingface.py
   ```

2. **Clear Streamlit cache**:
   - Option A: In app, press 'C' key, click "Clear cache"
   - Option B: Restart app from dashboard
   - Option C: Delete `.cache/` folder in repo

3. **App auto-updates**: No code changes needed!

### Update Code

Any push to `main` branch auto-redeploys:
```bash
git add .
git commit -m "Update feature X"
git push origin main
# Streamlit Cloud automatically redeploys
```

### Monitor Logs

View real-time logs:
1. Streamlit Cloud dashboard
2. Click your app
3. Click "Manage app" → "Logs"
4. See live application logs

---

## Cost Breakdown

### Free Tier (Currently Using)
- ✅ 1 public app
- ✅ 1GB RAM
- ✅ 1 CPU core
- ✅ Unlimited viewers
- ✅ Community support
- ❌ No private apps
- ❌ Limited compute

**Your app fits free tier perfectly!**

### Pro Tier ($20/month)
- 3 apps
- 3GB RAM per app
- Priority compute
- Private apps
- Email support

Only needed if:
- You want private deployment
- Need more concurrent users
- Want faster processing

---

## Security Notes

### Public Deployment
- ✅ Model files public on Hugging Face (intended)
- ✅ Source code public on GitHub (open source)
- ✅ No user data stored
- ✅ Images processed in memory only
- ✅ No authentication needed (educational tool)

### Private Deployment
If you need private deployment:
1. Make GitHub repo private
2. Make Hugging Face repo private
3. Add HF token to Streamlit secrets:
   ```toml
   HUGGINGFACE_TOKEN = "hf_your_token_here"
   ```
4. Update app.py to use token
5. Upgrade to Streamlit Pro tier

---

## Example Deployments

### Your Deployment
**Live URL**: https://mu-niu13-x-ray-transparency-lab-app-y74grj.streamlit.app/

**Configuration**:
- Repository: `Mu-niu13/X-Ray-Transparency-Lab`
- HF Model: `Mu-niu13/xray-pneumonia-model`
- Python: 3.10
- Tier: Free

### Custom URL Structure
```
https://[username]-[repo-name]-[app-name]-[random].streamlit.app/
```

---

## FAQ

**Q: How long does first deployment take?**
A: 8-12 minutes (build + model download)

**Q: How long do subsequent starts take?**
A: <30 seconds (cached files)

**Q: Can I use my own domain?**
A: Yes, with Pro tier

**Q: Is my deployment secure?**
A: Yes, served over HTTPS, no data stored

**Q: How many users can I support?**
A: Free tier: 1-2 concurrent users comfortably

**Q: Can I deploy multiple versions?**
A: Yes, create separate branches and deploy each

**Q: What if Streamlit Cloud goes down?**
A: Deploy to alternative platforms (see below)

---

## Alternative Deployment Options

### Heroku
```bash
# Requires different config
heroku create xray-transparency-lab
git push heroku main
```

### AWS/Azure/GCP
- Use Docker container
- Deploy to cloud compute service
- More complex but more control

### Self-Hosted
```bash
# Run on your own server
streamlit run app.py --server.port 8501
```

---

## Support

### Getting Help

1. **Streamlit Community**: https://discuss.streamlit.io/
2. **GitHub Issues**: https://github.com/Mu-niu13/X-Ray-Transparency-Lab/issues
3. **Hugging Face Forums**: https://discuss.huggingface.co/

### Report Bugs

1. Check existing issues first
2. Provide:
   - Error message
   - Steps to reproduce
   - Browser/OS info
   - Streamlit Cloud logs

---

## Checklist

Before going live, verify:

- [ ] Model uploaded to Hugging Face
- [ ] GitHub repo updated
- [ ] requirements.txt complete
- [ ] Secrets configured
- [ ] Python version = 3.10
- [ ] Test with sample images
- [ ] Test with uploaded images
- [ ] All tabs work
- [ ] Report generates
- [ ] README has correct link
- [ ] Share your deployment!

---

## Success! 🎉

Your app is now live at:
**https://mu-niu13-x-ray-transparency-lab-app-y74grj.streamlit.app/**

Share it with:
- Classmates/colleagues
- Research community
- On social media
- In your portfolio
