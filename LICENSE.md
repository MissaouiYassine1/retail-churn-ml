MIT License

Copyright (c) 2026 Yassine Missaoui

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Additional Notes

### Project Scope

This license applies to all source code, scripts, notebooks, and documentation
contained in the **Retail Customer Churn Prediction** project, including but
not limited to:

- Machine learning pipeline (`src/`)
- REST API (`app/api.py`)
- Streamlit dashboards (`app/app.py`, `app/app_api.py`)
- Utility and preprocessing modules (`src/utils.py`, `src/preprocessing.py`)
- Configuration and setup scripts (`mk-venv.ps1`, `requirements.txt`)

### Data & Model Artifacts

The **datasets** (`data/`) and **trained model artifacts** (`models/`) are
**not covered** by this license. They are proprietary and may not be
redistributed or used outside the scope of this project without explicit
written permission from the author.

### Third-Party Libraries

This project depends on third-party open-source libraries. Each library is
governed by its own license:

| Library | License |
|---|---|
| scikit-learn | BSD 3-Clause |
| XGBoost | Apache 2.0 |
| pandas | BSD 3-Clause |
| numpy | BSD 3-Clause |
| FastAPI | MIT |
| Streamlit | Apache 2.0 |
| Pydantic | MIT |
| Uvicorn | BSD 3-Clause |
| Matplotlib | PSF (BSD-compatible) |
| Seaborn | BSD 3-Clause |
| joblib | BSD 3-Clause |
| imbalanced-learn | MIT |

### Contact

For permissions beyond the scope of this license, please contact:

**Yassine Missaoui**
📧 yassine.missaoui@enis.tn