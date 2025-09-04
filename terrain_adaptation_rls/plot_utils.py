import matplotlib.pyplot as plt

def load_model(model_type, device, n_basis=8, path=None):
    match model_type:
        case "neural_ode":
            from models.neural_ode import load_model, loss_fn
            model = load_model(device=device, path=path, n_basis=n_basis).to(device)
        case "function_encoder":
            from models.function_encoder import load_model, loss_fn
            model = load_model(device=device, path=path, n_basis=n_basis).to(device)
        case "maml":
            from models.maml import load_model, loss_fn
            model = load_model(device=device, path=path, n_basis=n_basis).to(device)
        case _:
            raise ValueError(f"Unknown model type: {model_type}")
    return model, loss_fn

def format_fig():
    # Use STIX fonts (LaTeX-style) and apply them consistently
    plt.rcParams.update({
        'font.family': 'STIXGeneral',
        'mathtext.fontset': 'stix',
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
    })

    # Plotting
    fig = plt.figure(figsize=(3.5, 2.5))
    colors = {"neural_ode": "#D62728", "function_encoder": "#1F77B4", "rls": "#2ca02c", "maml": "#A200FF"}
    names = {"neural_ode": "NODE", "function_encoder": "FE", "rls": "FE-RLS", "maml": "MAML"}

    return fig, colors, names