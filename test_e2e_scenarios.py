import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
from unittest.mock import patch
import app

def test_scenario_a():
    print("[E2E SCENARIO A] Case Navigation Flow...")
    # 1. Start Overview
    with patch.object(app, 'ctx') as mock_ctx:
        mock_ctx.triggered = [{'prop_id': 'nav-btn-overview.n_clicks', 'value': 1}]
        mock_ctx.triggered_id = 'nav-btn-overview'
        v, g, p = app.handle_view_navigation(0, 1, 0, 0, 0, 0, 0, [0])
        assert v == 'overview'

    # 2. Open Case 1
    with patch.object(app, 'ctx') as mock_ctx:
        mock_ctx.triggered = [{'prop_id': '{"index":"FRAUD_SMURFING_FUNNEL","type":"open-workspace-btn"}.n_clicks', 'value': 1}]
        mock_ctx.triggered_id = {'index': 'FRAUD_SMURFING_FUNNEL', 'type': 'open-workspace-btn'}
        c, v, g, s, tab, p, focal = app.handle_open_workspace_btn([1])
        assert v == 'workspace'
        assert c == 'FRAUD_SMURFING_FUNNEL'
        assert focal == 'ACC_FRAUD_MASTER'

    # 3. Switch Tabs
    for target_tab in ['network', 'why', 'xai', 'report', 'summary']:
        with patch.object(app, 'ctx') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': f'{{"tab":"{target_tab}","type":"ws-tab-btn"}}.n_clicks', 'value': 1}]
            mock_ctx.triggered_id = {'tab': target_tab, 'type': 'ws-tab-btn'}
            t = app.handle_workspace_tabs([1])
            assert t == target_tab

    # 4. Toggle Explain Guide
    with patch.object(app, 'ctx') as mock_ctx:
        mock_ctx.triggered = [{'prop_id': '{"action":"toggle-guide","type":"guide-mode-btn"}.n_clicks', 'value': 1}]
        mock_ctx.triggered_id = {'action': 'toggle-guide', 'type': 'guide-mode-btn'}
        g, p, s = app.handle_guide_mode_toggle([1], False, False)
        assert g == True
        assert s == 1

    # 5. Go to Report
    rep_view = app.render_report_view('FRAUD_SMURFING_FUNNEL')
    assert rep_view is not None

    # 6. Logo click to Overview
    with patch.object(app, 'ctx') as mock_ctx:
        mock_ctx.triggered = [{'prop_id': 'nav-brand-logo.n_clicks', 'value': 1}]
        mock_ctx.triggered_id = 'nav-brand-logo'
        v, g, p = app.handle_view_navigation(1, 0, 0, 0, 0, 0, 0, [0])
        assert v == 'overview'
        assert g == False
        assert p == False

    print(" -> Scenario A Passed: Complete navigation lifecycle verified.")

def test_scenario_b():
    print("[E2E SCENARIO B] Case Isolation & No Cross-Contamination...")
    meta_1 = app.SCENARIOS['FRAUD_SMURFING_FUNNEL']
    meta_2 = app.SCENARIOS['FRAUD_SYBIL_DEVICE_RING']
    assert meta_1['focal_node'] != meta_2['focal_node']
    
    # SHAP isolation
    chart_1 = app.build_shap_chart(meta_1['focal_node'])
    chart_2 = app.build_shap_chart(meta_2['focal_node'])
    assert chart_1 is not None and chart_2 is not None
    
    # Graph element isolation
    elem_1 = app.get_subgraph_elements('FRAUD_SMURFING_FUNNEL')
    elem_2 = app.get_subgraph_elements('FRAUD_SYBIL_DEVICE_RING')
    nodes_1 = {el['data']['id'] for el in elem_1 if 'source' not in el['data']}
    nodes_2 = {el['data']['id'] for el in elem_2 if 'source' not in el['data']}
    assert meta_1['focal_node'] in nodes_1
    assert meta_2['focal_node'] in nodes_2
    assert meta_1['focal_node'] not in nodes_2
    print(" -> Scenario B Passed: Case 1 and Case 2 strictly isolated.")

def test_scenario_c():
    print("[E2E SCENARIO C] Presentation Mode Step Progression & Controls...")
    # Enter Presentation Mode
    with patch.object(app, 'ctx') as mock_ctx:
        mock_ctx.triggered = [{'prop_id': '{"action":"toggle-presentation","type":"guide-mode-btn"}.n_clicks', 'value': 1}]
        mock_ctx.triggered_id = {'action': 'toggle-presentation', 'type': 'guide-mode-btn'}
        g, p, s = app.handle_guide_mode_toggle([1], False, False)
        assert g == True and p == True and s == 1

    # Advance steps
    current_step = 1
    for step_target in range(2, 7):
        with patch.object(app, 'ctx') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': '{"direction":"next","type":"guide-step-btn"}.n_clicks', 'value': 1}]
            mock_ctx.triggered_id = {'direction': 'next', 'type': 'guide-step-btn'}
            current_step = app.handle_guide_step_navigation([1], current_step, 'FRAUD_SMURFING_FUNNEL')
            assert current_step == step_target

    # Reverse step
    with patch.object(app, 'ctx') as mock_ctx:
        mock_ctx.triggered = [{'prop_id': '{"direction":"prev","type":"guide-step-btn"}.n_clicks', 'value': 1}]
        mock_ctx.triggered_id = {'direction': 'prev', 'type': 'guide-step-btn'}
        current_step = app.handle_guide_step_navigation([1], current_step, 'FRAUD_SMURFING_FUNNEL')
        assert current_step == 5

    # Restart
    with patch.object(app, 'ctx') as mock_ctx:
        mock_ctx.triggered = [{'prop_id': '{"direction":"restart","type":"guide-step-btn"}.n_clicks', 'value': 1}]
        mock_ctx.triggered_id = {'direction': 'restart', 'type': 'guide-step-btn'}
        current_step = app.handle_guide_step_navigation([1], current_step, 'FRAUD_SMURFING_FUNNEL')
        assert current_step == 1

    # Exit
    with patch.object(app, 'ctx') as mock_ctx:
        mock_ctx.triggered = [{'prop_id': '{"action":"exit-presentation","type":"guide-mode-btn"}.n_clicks', 'value': 1}]
        mock_ctx.triggered_id = {'action': 'exit-presentation', 'type': 'guide-mode-btn'}
        g, p, s = app.handle_guide_mode_toggle([1], True, True)
        assert p == False
        assert g == False
        assert s == 1
    print(" -> Scenario C Passed: Presentation Mode fully verified across all 6 steps.")

def test_scenario_d():
    print("[E2E SCENARIO D] Report Generation, FIU-STR JSON & Cypher Export...")
    # STR JSON download
    with patch.object(app, 'ctx') as mock_ctx:
        mock_ctx.triggered = [{'prop_id': 'btn-download-str-json.n_clicks', 'value': 1}]
        mock_ctx.triggered_id = 'btn-download-str-json'
        d_res = app.download_str_file(1, 'FRAUD_SMURFING_FUNNEL')
        assert d_res is not None
        assert 'content' in d_res
        data = json.loads(d_res['content'])
        assert 'filing_institution' in data
        assert 'filing_reference' in data
        assert 'focal_subject' in data
        assert 'regulatory_statute' in data
        assert 'typology' in data
        assert 'algorithmic_narrative' in data
        assert 'FC-CASE' in data['case_id']

    # Cypher export download
    with patch.object(app, 'ctx') as mock_ctx:
        mock_ctx.triggered = [{'prop_id': 'btn-export-cypher-file.n_clicks', 'value': 1}]
        mock_ctx.triggered_id = 'btn-export-cypher-file'
        c_res = app.download_cypher_file(1, 'FRAUD_SMURFING_FUNNEL')
        assert c_res is not None
        assert 'content' in c_res
        assert 'CREATE' in c_res['content'] or 'MERGE' in c_res['content']
    print(" -> Scenario D Passed: STR JSON and Cypher scripts verified.")

def test_scenario_e():
    print("[E2E SCENARIO E] Failure Recovery & Graceful Fallbacks...")
    # Nonexistent case key in workspace view
    ws_view = app.render_workspace_view('NON_EXISTENT_KEY_9999', 'summary', False, 1, False, 'concentric', 'ALL')
    assert ws_view is not None

    # Nonexistent case key in report view
    rep_view = app.render_report_view('NON_EXISTENT_KEY_9999')
    assert rep_view is not None

    # Unknown node tap in graph
    inspector, sel_node = app.handle_graph_node_click({'id': 'UNKNOWN_NODE_ID_1234'}, 'FRAUD_SMURFING_FUNNEL')
    assert inspector is not None
    assert "not found" in str(inspector)
    print(" -> Scenario E Passed: Graceful recovery and fallbacks verified.")

if __name__ == '__main__':
    print("============================================================")
    print(" FRAUDCIRCLE E2E SCENARIOS A THROUGH E")
    print("============================================================")
    test_scenario_a()
    test_scenario_b()
    test_scenario_c()
    test_scenario_d()
    test_scenario_e()
    print("============================================================")
    print(" ALL E2E SCENARIOS VALIDATED SUCCESSFULLY!")
    print("============================================================")
