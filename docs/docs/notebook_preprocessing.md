# Notebook EDA

The notebook is here as a space to preprocessing data.

**Project goal :** using magnetic resonance imaging (MRI) data, detect the presence of tumors and classify them.

**Metrics :**
F1 score + recall.

I prefer to use their metrics compared to accuracy because the objective is early detection, which allows for monitoring and further examinations. It's more important to focus on avoiding false negatives, even if the accuracy isn't optimal.

## Preprocessing

**Goal :** Create a pipeline that performs the entire fixed preprocessing and saves the new data in the 'processed' directory.