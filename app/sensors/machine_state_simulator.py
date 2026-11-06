"""
app/sensors/machine_state_simulator.py

Simule l'état marche/arrêt de la machine, en l'absence d'accès direct aux
capteurs/PLC du site OCP dans le délai du POC (cf. cahier des charges,
section 5 et 12 : "l'état marche/arrêt de la machine sera simulé").

Principe de précaution (aligné sur le standard S-HSE-22, consignation) :
    "Toute installation est considérée en service et donc dangereuse, tant
    que la vérification de l'élimination des sources d'énergie n'a pas été
    réalisée."
Donc par défaut, si l'état n'est pas explicitement confirmé "arrêt", on
considère la machine comme EN MARCHE (comportement prudent).
"""


class MachineStateSimulator:
    """
    Simule l'état d'une ou plusieurs machines. Pour le POC, l'état est fixé
    manuellement (pas de vrai capteur/PLC connecté) — mais l'interface est
    conçue pour qu'un vrai capteur puisse remplacer cette simulation plus
    tard sans changer le reste du code (même méthode is_running()).

    Utilisation simple (une seule machine, état global) :
        simulator = MachineStateSimulator(default_state="marche")
        simulator.is_running()  # True

    Utilisation avancée (état différent par zone/machine) :
        simulator.set_state("convoyeur_A", "arret")
        simulator.is_running("convoyeur_A")  # False
        simulator.is_running("convoyeur_B")  # True (défaut, non confirmé arrêté)
    """

    VALID_STATES = {"marche", "arret", "inconnu"}

    def __init__(self, default_state: str = "inconnu", unknown_is_running: bool = True):
        """
        Args:
            default_state: état par défaut ("marche", "arret" ou "inconnu") appliqué
                à toute machine dont l'état n'a pas été fixé explicitement.
            unknown_is_running: applique une politique prudente aux alertes
                de proximité si l'état de la machine est inconnu.
        """
        if default_state not in self.VALID_STATES:
            raise ValueError(f"default_state invalide : {default_state}")
        self.default_state = default_state
        self.unknown_is_running = unknown_is_running
        self._states: dict[str, str] = {}

    def set_state(self, machine_id: str, state: str):
        """Fixe manuellement l'état d'une machine identifiée par machine_id."""
        if state not in self.VALID_STATES:
            raise ValueError(f"state invalide : {state}")
        self._states[machine_id] = state

    def is_running(self, machine_id: str = "default") -> bool:
        """
        Retourne True si la machine est considérée en marche.
        Un état inconnu peut être traité comme actif par politique de prudence.
        """
        state = self._states.get(machine_id, self.default_state)
        return state == "marche" or (state == "inconnu" and self.unknown_is_running)

    def get_state_label(self, machine_id: str = "default") -> str:
        """Retourne l'état observé ('marche', 'arret' ou 'inconnu')."""
        return self._states.get(machine_id, self.default_state)
