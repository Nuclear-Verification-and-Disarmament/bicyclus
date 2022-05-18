"""Extract simulation results from a Cyclus SQLite simulation database."""

import sqlite3


def run_with_conn(filename, extract, params={}):
    """Open a connection to an sqlite an perform the `extract` query.

    Parameters
    ----------
    filename : str
    extract : function

    params : dict
        Additional parameters passed to `extract`.
    """
    with sqlite3.connect(filename) as sql:
        return extract(sql, **params)

def get_all_agents(sqlite):
    query = """
SELECT AgentId, Kind, Spec, Prototype
FROM AgentEntry
ORDER BY AgentId ASC"""
    cursor = sqlite.execute(query)

    out = []
    for row in cursor:
        out.append('{} {} spec {} proto {}'.format(*row))
    return '\n'.join(out)

def multi_agent_concs(fname, sinks):
    """Query concentrations and masses of multiple sinks.

    Parameters
    ----------
    fname : str
        Name of the last sqlite file.
    sinks : list of str
        Each entry is the name of a sink present in the Cyclus sqlite file.

    Returns
    -------
    all_concentrations : dict
        Contains the concentrations (normalised to 1) of the sinks as values
        and the sink names (as defined in `sinks`) as keys.
    masses : dict
        Contains the masses of the material in the sinks as values and the
        sink names (as defined in `sinks`) as keys.
    """
    all_concentrations = {}
    masses = {}
    for sink in sinks:
        concentrations, mass = run_with_conn(fname,
                                             extract_isotope_concentrations,
                                             {'agent_name': sink})
        all_concentrations[sink] = concentrations
        masses[sink] = mass
    return all_concentrations, masses

def multi_agent_transactions(fname, sinks, mass_sinks=[]):
    """Query concentrations and masses of multiple sinks.

    Conceptually identical to `multi_agent_concs`, but uses
    `extract_transaction_composition` which is a method tailor-suited to the
    Pakistan scenario.

    Parameters
    ----------
    fname : str
        Name of the last sqlite file.
    sinks : list of str
        Each entry is the name of a sink present in the Cyclus sqlite file.
    mass_sinks : list of str, optional
        Each entry is the name of a sink present in the Cyclus sqlite file. If
        set, the final mass of each of the sinks is added to the returned
        `masses`. Note that this uses a non-official Cycamore feature, see
        https://github.com/maxschalz/cycamore/commit/cdda33e3704e93ab37b8a2218da15a01b0d80155

    Returns
    -------
    all_concentrations : dict(str, dict(int, float))
        Keys are the sink names, as defined in `sinks`, values are composition
        dicts as returned by `extract_transaction_composition`.
    masses : dict(str, float)
        Keys are the sink names, as defined in `sinks` and `mass_sinks`, values
        are the masses of the last transactions performed to these sinks.
    """
    all_concentrations = {}
    masses = {}
    for sink in sinks:
        concentrations, mass = run_with_conn(
            fname, extract_transaction_composition, {"agent_name": sink})
        all_concentrations[sink] = concentrations
        masses[sink] = mass

    for mass_sink in mass_sinks:
        mass = run_with_conn(fname, extract_mass, {"agent_name": mass_sink})
        masses[mass_sink] = mass

    return all_concentrations, masses

def extract_mass(sqlite, agent_name):
    """Extract the mass at the end of the simulation from sink `agent_name`.

    Note that this uses a non-official Cycamore feature, see
    https://github.com/maxschalz/cycamore/commit/bbfa3856dc83a89a2f198ca624e040d729b0b399

    Parameters
    ----------
    sqlite : an open Sqlite3 connection
    agent_name : str
        Name of the agent from where to extract the mass. Note that this
        facility name should be unique (i.e., don't build identically-named
        sinks).

    Returns
    -------
    mass : float
    """
    agent_id_query = """
SELECT AgentId
FROM AgentEntry
WHERE Prototype = :agentname"""
    cursor = sqlite.execute(agent_id_query, {"agentname": agent_name})
    results = cursor.fetchall()
    # Cursor will only contain one element *unless* facility has been built
    # multiple times.
    if len(results) != 1:
        msg = (f"Multiple or zero instances of agent named '{agent_name}' "
                "present in `AgentEntry`.")
        raise ValueError(msg)
    agent_id = results[0][0]

    # Requires cherry-picked cycamore version, see docstring of this function.
    mass_query = """
SELECT Value
FROM TimeSeriesSinkTotalMats
WHERE AgentId = :agentid AND Time = (SELECT MAX(Time)
                                     FROM TimeSeriesSinkTotalMats
                                     WHERE AgentId = :agentid);
"""
    cursor = sqlite.execute(mass_query, {"agentid": agent_id})
    results = cursor.fetchall()
    if len(results) != 1:
        msg = (f"Multiple or zero entries of material of agent named "
               f"'{agent_name}' stored in `TimeSeriesSinkTotalMats`.")
        raise ValueError(msg)
    mass = results[0][0]
    return mass

def extract_transaction_composition(sqlite, agent_name):
    """Get the composition of the last transaction to `agent_name`.

    This is *not* a very general function but rather tailor-suited to the
    Pakistan scenario. `extract_transaction_concentration` cannot be used
    because the `ExplicitInventory` option is turned off to reduce runtime (by
    two magnitudes --> worth it).

    Parameters
    ----------
    sqlite : an open Sqlite3 connection
    agent_name : str
        Name of the agent from where to extract the composition. Note that this
        facility name should be unique (i.e., don't build 5 sinks named
        "Sink").

    Returns
    -------
    composition : dict(int, float)
        Keys are the nuc-ids (e.g., 92235, 55137, ...) and values are the
        *mass* fractions, normalised to 1.

    quantity : float
        The material's mass of the last transaction (the one considered here).
    """
    agent_id_query = """
SELECT AgentId
FROM AgentEntry
WHERE Prototype = :agentname"""
    cursor = sqlite.execute(agent_id_query, {"agentname": agent_name})
    results = cursor.fetchall()
    # Cursor will only contain one element *unless* facility has been built
    # multiple times.
    if len(results) != 1:
        msg = (f"Multiple or zero instances of agent named '{agent_name}' "
                "present in `AgentEntry`.")
        raise ValueError(msg)
    agent_id = results[0][0]

    resource_id_query = """
SELECT ResourceId
FROM Transactions
WHERE ReceiverId = :agentid"""
    cursor = sqlite.execute(resource_id_query, {"agentid": agent_id})
    results = cursor.fetchall()
    if len(results) != 1:
        msg = (f"Multiple or zero transactions to agent named '{agent_name}' "
                "stored in `Transactions`.")
        raise ValueError(msg)
    resource_id = results[0][0]

    mass_qual_id_query = """
SELECT QualId, Quantity
FROM Resources
WHERE ResourceId = :resourceid"""
    cursor = sqlite.execute(mass_qual_id_query, {"resourceid": resource_id})
    qual_id, quantity = cursor.fetchone()  # Resource Ids are unique.

    composition_query = """
SELECT NucId, MassFrac
FROM Compositions
WHERE QualId = :qualid"""
    cursor = sqlite.execute(composition_query, {"qualid": qual_id})
    composition = {}
    # Cyclus does normalise the compositions before storing it, I
    # double-checked this (as of Jan. 2022).
    for row in cursor:
        composition[row[0]] = row[1]

    return composition, quantity

def extract_isotope_concentrations(sqlite,
                                   agent_name='DepletedUraniumSink',
                                   inventory_name='inventory'):
    """Select the most recent composition for a given agent and inventory.

    The composition is normalised to 1.
    Returns
    -------
    (isotopes, total_mass) : tuple
        `isotopes` is a dict with the isotope IDs (int, not str) as keys and
        the corresponding fractions as values. The fractions sum up to 1.
        `total_mass` is the total mass of the material in the agent's
        inventory.
    """
    agentidquery = """
SELECT AgentId
FROM AgentEntry
WHERE Prototype = :agentname"""
    cursor = sqlite.execute(agentidquery, {'agentname': agent_name})
    agentid = -1
    for row in cursor:
        agentid = int(row[0])

    isoquery = """
SELECT NucId, Quantity
FROM ExplicitInventory
WHERE AgentId = :agentid AND InventoryName = :invname
AND Time = (SELECT MAX(Time)
            FROM ExplicitInventory
            WHERE AgentId = :agentid);
"""
    cursor = sqlite.execute(isoquery, {
        'agentid': agentid,
        'invname': inventory_name
    })

    isotopes = {}
    for row in cursor:
        isotopes[row[0]] = row[1]

    total_mass = sum(v for (k, v) in isotopes.items())
    for (k, v) in isotopes.items():
        isotopes[k] = v / total_mass
    return isotopes, total_mass
