# Streamlit Community Cloud: open https://share.streamlit.io
# 1. Sign in with GitHub (inezaodon)
# 2. New app → repo: inezaodon/image-quality-cnn
# 3. Branch: main · Main file: streamlit_app.py
# 4. Deploy
#
# Local test:  python3 -m streamlit run streamlit_app.py
#
# Redeploys / URL behavior
# ------------------------
# Streamlit Cloud watches the connected GitHub branch. When you push to that
# branch it rebuilds the SAME app URL (it does not mint a new link). Turn on
# "Automatic updates" / "Main file" watching in the app settings if pushes do
# not redeploy by themselves; or click "Reboot app" / "Rerun" after a push.
#
# PDFs
# ----
# Do not link GitHub /blob/… pages for PDFs — GitHub does not render them.
# The app serves downloads from the repo files and opens PDFs in Mozilla’s
# pdf.js viewer pointed at raw.githubusercontent.com.
