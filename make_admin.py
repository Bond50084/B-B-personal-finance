from app import app, db, User

EMAIL = "info@behrens-behrens.org"   # ← change to the email you registered with

with app.app_context():
    u = User.query.filter_by(email=EMAIL).first()
    if u is None:
        print(f"No user found with email {EMAIL!r}. Register first, then re-run.")
    else:
        u.is_admin = True
        u.has_license = True   # also unblocks the simulator for you
        db.session.commit()
        print(f"Done. {EMAIL} → is_admin={u.is_admin}, has_license={u.has_license}")
