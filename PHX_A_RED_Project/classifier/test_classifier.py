from PHX_A_RED_Project.classifier import AREDClassifier
from PHX_A_RED_Project.data.base_stream import PerchDataStream

# Lower kappa => more queries => more prototypes (trade-off: more labels used)
clf = AREDClassifier(
    kappa=0.35,
    data_window_size=4000,
    n_neighbors=1,
    # smart_forgetting_var=(1, 0.05),  # optional: try AREDIN smart forgetting mode
)

clf.fit(
    num_points=12000,                 # Start with a few thousand for speed
    label_column="scientific_name",
    train_frac=0.75
)

print("Support info:", clf.get_support_info())

# Test stream (independent)
test_stream = PerchDataStream(
    label_column="scientific_name",
    shuffle=True,
    seed=99,
    max_samples=4000
)

metrics = clf.evaluate(test_stream)