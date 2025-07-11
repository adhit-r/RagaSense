import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import sqlalchemy
import pandas as pd
import os
import shutil
from datetime import datetime
from ml.raga_classifier import RagaClassifier, extract_features_from_file
import mlflow
from werkzeug.utils import secure_filename

# --- Config ---
AUDIO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'audio_uploads'))
MLFLOW_TRACKING_URI = 'sqlite:///mlflow.db'  # or your mlflow server
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# --- DB Setup ---
engine = sqlalchemy.create_engine('postgresql://user:password@localhost:5432/yourdb')

def get_audio_samples():
    df = pd.read_sql('SELECT * FROM audio_samples ORDER BY id DESC', engine)
    return df

def update_label(audio_id, label):
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text('UPDATE audio_samples SET raga_label=:label WHERE id=:id'), {'label': label, 'id': audio_id})

def insert_audio(file_path):
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text('INSERT INTO audio_samples (file_path) VALUES (:fp)'), {'fp': file_path})

def allowed_file(filename):
    return filename.lower().endswith('.mp3')

# --- Dash App ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H2("Raga ML Trainer"),
    html.Hr(),
    html.H4("Audio Samples"),
    dcc.Upload(
        id='upload-audio',
        children=html.Div(['Drag and Drop or ', html.A('Select MP3 Files')]),
        style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center'},
        multiple=True
    ),
    html.Div(id='upload-status'),
    dash_table.DataTable(
        id='audio-table',
        columns=[
            {'name': 'ID', 'id': 'id'},
            {'name': 'File', 'id': 'file_path'},
            {'name': 'Raga Label', 'id': 'raga_label'},
        ],
        row_selectable='multi',
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
    ),
    html.Br(),
    html.Audio(id='audio-player', controls=True, src=''),
    html.Br(),
    dcc.Input(id='label-input', type='text', placeholder='Enter raga label'),
    html.Button('Update Label', id='update-label-btn'),
    html.Div(id='label-status'),
    html.Hr(),
    html.H4("Train Model"),
    dcc.Dropdown(
        id='model-type',
        options=[
            {'label': 'Dense NN', 'value': 'dense'},
            {'label': 'SVM', 'value': 'svm'},
            {'label': 'Logistic Regression', 'value': 'logreg'},
            {'label': 'XGBoost', 'value': 'xgb'},
        ],
        value='dense'
    ),
    html.Button('Train', id='train-btn'),
    html.Div(id='train-output'),
    html.Hr(),
    html.H4("Results"),
    html.Div(id='results-output'),
])

# --- Callbacks ---

@app.callback(
    Output('audio-table', 'data'),
    Input('upload-status', 'children'),
    Input('label-status', 'children'),
    Input('train-btn', 'n_clicks'),
)
def refresh_table(u, l, t):
    return get_audio_samples().to_dict('records')

@app.callback(
    Output('audio-player', 'src'),
    Input('audio-table', 'selected_rows'),
    State('audio-table', 'data')
)
def play_audio(selected, data):
    if selected and data:
        file_path = data[selected[0]]['file_path']
        if os.path.exists(file_path):
            return '/assets/' + os.path.basename(file_path)
    return ''

@app.callback(
    Output('upload-status', 'children'),
    Input('upload-audio', 'contents'),
    State('upload-audio', 'filename')
)
def upload_audio(contents, filenames):
    if contents and filenames:
        for content, filename in zip(contents, filenames):
            if not allowed_file(filename):
                return f"Only MP3 files allowed: {filename}"
            fname = secure_filename(filename)
            save_path = os.path.join(AUDIO_DIR, fname)
            # Save file
            content_string = content.split(',')[1]
            import base64
            with open(save_path, 'wb') as f:
                f.write(base64.b64decode(content_string))
            insert_audio(save_path)
        return f"Uploaded {len(filenames)} file(s)."
    return ''

@app.callback(
    Output('label-status', 'children'),
    Input('update-label-btn', 'n_clicks'),
    State('audio-table', 'selected_rows'),
    State('audio-table', 'data'),
    State('label-input', 'value')
)
def update_label_callback(n, selected, data, label):
    if n and selected and label:
        audio_id = data[selected[0]]['id']
        update_label(audio_id, label)
        return f"Label updated for ID {audio_id}."
    return ''

@app.callback(
    Output('train-output', 'children'),
    Output('results-output', 'children'),
    Input('train-btn', 'n_clicks'),
    State('audio-table', 'data'),
    State('model-type', 'value')
)
def train_model(n, data, model_type):
    if not n:
        return '', ''
    # Filter labeled samples
    df = pd.DataFrame(data)
    df = df[df['raga_label'].notnull()]
    X, y = [], []
    for _, row in df.iterrows():
        feats = extract_features_from_file(row['file_path'])
        X.extend(feats)
        y.extend([row['raga_label']] * len(feats))
    X, y = np.array(X), np.array(y)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    clf = RagaClassifier(model_type=model_type, num_classes=len(le.classes_))
    import mlflow
    with mlflow.start_run(run_name=f"{model_type}_{datetime.now().isoformat()}"):
        mlflow.log_param('model_type', model_type)
        mlflow.log_param('num_classes', len(le.classes_))
        if model_type == 'dense' and len(le.classes_) > 2:
            from tensorflow.keras.utils import to_categorical
            y_cat = to_categorical(y_enc, len(le.classes_))
            clf.train(X, y_cat)
        else:
            clf.train(X, y_enc)
        # Save model
        model_path = f"mlflow_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        clf.save(model_path)
        mlflow.log_artifact(model_path)
        # Evaluate
        y_pred = [clf.predict([x]) for x in X]
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        acc = accuracy_score(y_enc, y_pred)
        mlflow.log_metric('accuracy', acc)
        cm = confusion_matrix(y_enc, y_pred)
        report = classification_report(y_enc, y_pred, target_names=le.classes_)
        mlflow.log_text(report, 'classification_report.txt')
    return f"Training complete. Accuracy: {acc:.3f}", html.Pre(report)

if __name__ == '__main__':
    if not os.path.exists(AUDIO_DIR):
        os.makedirs(AUDIO_DIR)
    app.run_server(debug=True) 