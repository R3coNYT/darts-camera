#!/usr/bin/env bash
# DartsCamera – Script de mise à jour (Linux / macOS / Raspberry Pi)

set -Eeuo pipefail

# ── Couleurs ───────────────────────────────────────────────────────────────────

COLOR_RED="\033[1;31m"
COLOR_GREEN="\033[1;32m"
COLOR_YELLOW="\033[1;33m"
COLOR_BLUE="\033[1;34m"
COLOR_CYAN="\033[1;36m"
COLOR_RESET="\033[0m"

log()  { echo -e "${COLOR_BLUE}[+]${COLOR_RESET} $*"; }
ok()   { echo -e "${COLOR_GREEN}[✓]${COLOR_RESET} $*"; }
warn() { echo -e "${COLOR_YELLOW}[!]${COLOR_RESET} $*"; }
err()  { echo -e "${COLOR_RED}[✗]${COLOR_RESET} $*" >&2; }
info() { echo -e "${COLOR_CYAN}[i]${COLOR_RESET} $*"; }

cleanup_on_error() {
    err "La mise à jour a échoué à la ligne $1"
    err "Vos fichiers de configuration n'ont PAS été modifiés."
    exit 1
}
trap 'cleanup_on_error $LINENO' ERR

# ── Localiser le dossier d'installation ───────────────────────────────────────

find_install_dir() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -f "$script_dir/app.py" ] && [ -d "$script_dir/.git" ]; then
        INSTALL_DIR="$script_dir"
        return
    fi

    if [ -f "$(pwd)/app.py" ] && [ -d "$(pwd)/.git" ]; then
        INSTALL_DIR="$(pwd)"
        return
    fi

    err "Impossible de localiser le dossier DartsCamera."
    err "Lancez ce script depuis le dossier d'installation."
    exit 1
}

# ── Sauvegarde de la config ────────────────────────────────────────────────────

backup_config() {
    local timestamp
    timestamp="$(date +%Y%m%d_%H%M%S)"
    BACKUP_DIR="$INSTALL_DIR/backups/$timestamp"

    log "Sauvegarde de la config → $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"

    if [ -f "$INSTALL_DIR/config.yaml" ]; then
        cp "$INSTALL_DIR/config.yaml" "$BACKUP_DIR/config.yaml"
        ok "config.yaml sauvegardé"
    else
        warn "Aucun config.yaml trouvé — rien à sauvegarder"
    fi

    if [ -f "$INSTALL_DIR/.env" ]; then
        cp "$INSTALL_DIR/.env" "$BACKUP_DIR/.env"
        ok ".env sauvegardé"
    fi

    ok "Sauvegarde terminée → $BACKUP_DIR"
}

# ── Git pull ───────────────────────────────────────────────────────────────────

git_update() {
    log "Récupération du dernier code depuis GitHub"
    cd "$INSTALL_DIR"

    if ! command -v git &>/dev/null; then
        err "git n'est pas installé. Installez-le avec votre gestionnaire de paquets."
        exit 1
    fi

    # Mettre en stash les modifications locales sur les fichiers suivis
    local stash_result
    stash_result="$(git stash 2>&1)" || true
    if ! echo "$stash_result" | grep -q "No local changes"; then
        info "Modifications locales mises en stash : $stash_result"
    fi

    git fetch origin

    local branch
    branch="$(git rev-parse --abbrev-ref HEAD)"
    git reset --hard "origin/$branch"

    # Restaurer config.yaml depuis la sauvegarde (ne pas l'écraser avec la version dépôt)
    if [ -f "$BACKUP_DIR/config.yaml" ]; then
        cp -f "$BACKUP_DIR/config.yaml" "$INSTALL_DIR/config.yaml"
        ok "config.yaml restauré (vos réglages sont préservés)"
    fi

    if [ -f "$BACKUP_DIR/.env" ]; then
        cp -f "$BACKUP_DIR/.env" "$INSTALL_DIR/.env"
        ok ".env restauré"
    fi

    local new_commit
    new_commit="$(git rev-parse --short HEAD)"
    ok "Code mis à jour → commit $new_commit (branche : $branch)"
}

# ── Mise à jour des dépendances Python ────────────────────────────────────────

update_deps() {
    log "Mise à jour des dépendances Python"

    local venv_pip="$INSTALL_DIR/.venv/bin/pip"

    if [ ! -f "$venv_pip" ]; then
        warn "Venv introuvable (.venv). Création en cours..."
        python3 -m venv "$INSTALL_DIR/.venv"
        ok "Venv créé"
    fi

    "$venv_pip" install --upgrade pip --quiet
    "$venv_pip" install -r "$INSTALL_DIR/requirements.txt" --quiet
    ok "Dépendances Python à jour"
}

# ── Supprimer les anciennes sauvegardes (garder les 5 dernières) ───────────────

prune_backups() {
    local backup_root="$INSTALL_DIR/backups"
    [ -d "$backup_root" ] || return

    local count
    count="$(ls -1 "$backup_root" 2>/dev/null | wc -l)"

    if [ "$count" -gt 5 ]; then
        ls -1 "$backup_root" | sort | head -n $(( count - 5 )) | while read -r old; do
            rm -rf "$backup_root/$old"
            info "Ancienne sauvegarde supprimée : $old"
        done
    fi
}

# ── Main ───────────────────────────────────────────────────────────────────────

echo ""
echo -e "${COLOR_CYAN}====================================${COLOR_RESET}"
echo -e "${COLOR_CYAN}   DartsCamera — Mise à jour${COLOR_RESET}"
echo -e "${COLOR_CYAN}====================================${COLOR_RESET}"
echo ""

find_install_dir
info "Dossier d'installation : $INSTALL_DIR"

backup_config
git_update
update_deps
prune_backups

echo ""
echo -e "${COLOR_GREEN}====================================${COLOR_RESET}"
echo -e "${COLOR_GREEN}   Mise à jour terminée !${COLOR_RESET}"
echo -e "${COLOR_GREEN}====================================${COLOR_RESET}"
echo ""
info "Pour lancer l'application :"
echo "  source .venv/bin/activate"
echo "  python app.py"
echo ""
