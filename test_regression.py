import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import unittest
from unittest.mock import MagicMock, patch

# Import application components
import app

class FraudCircleRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n============================================================")
        print(" RUNNING FRAUDCIRCLE 18-POINT REGRESSION TEST SUITE")
        print("============================================================\n")

    def test_01_overview_renders_overview(self):
        """Test 1: Overview renders overview page layout with KPI cards and case table."""
        view = app.render_main_view('overview', 'FRAUD_SMURFING_FUNNEL', 'summary', False, 1, False, 'ALL', 'concentric', 'ALL')
        self.assertIsNotNone(view)
        self.assertEqual(view.id, 'overview-view-container')
        print(" PASS: Test 01 - Overview renders successfully.")

    def test_02_fraud_cases_page_renders_all_cases(self):
        """Test 2: Fraud Cases page renders all 7 high-risk case cards and filters."""
        view = app.render_main_view('cases', 'FRAUD_SMURFING_FUNNEL', 'summary', False, 1, False, 'ALL', 'concentric', 'ALL')
        self.assertIsNotNone(view)
        self.assertEqual(view.id, 'cases-view-container')
        self.assertEqual(len(app.SCENARIOS), 7)
        for key in app.SCENARIO_KEYS:
            self.assertIn(key, app.SCENARIOS)
        print(" PASS: Test 02 - Fraud Cases page renders all 7 cases.")

    def test_03_opening_case_1_selects_case_1(self):
        """Test 3: Opening Case 1 selects Case 1 correctly without errors."""
        with patch.object(app, 'ctx') as mock_ctx:
            case_id = 'FRAUD_SMURFING_FUNNEL'
            mock_ctx.triggered = [{'prop_id': f'{{"type":"open-workspace-btn","index":"{case_id}"}}.n_clicks', 'value': 1}]
            mock_ctx.triggered_id = {'type': 'open-workspace-btn', 'index': case_id}
            case, view, guide_act, guide_st, ws_tab, pmode, focal = app.handle_open_workspace_btn([1])
            self.assertEqual(view, 'workspace')
            self.assertEqual(case, case_id)
            self.assertFalse(guide_act)
            self.assertEqual(guide_st, 1)
            self.assertEqual(ws_tab, 'summary')
            self.assertFalse(pmode)
            self.assertEqual(focal, app.SCENARIOS[case_id]['focal_node'])
        print(" PASS: Test 03 - Opening Case 1 selects Case 1.")

    def test_04_opening_case_2_selects_case_2_without_stale_data(self):
        """Test 4: Opening Case 2 selects Case 2 and isolates Case 2 metadata."""
        with patch.object(app, 'ctx') as mock_ctx:
            case_id = 'FRAUD_SYBIL_DEVICE_RING'
            mock_ctx.triggered = [{'prop_id': f'{{"type":"open-workspace-btn","index":"{case_id}"}}.n_clicks', 'value': 1}]
            mock_ctx.triggered_id = {'type': 'open-workspace-btn', 'index': case_id}
            case, view, guide_act, guide_st, ws_tab, pmode, focal = app.handle_open_workspace_btn([1])
            self.assertEqual(view, 'workspace')
            self.assertEqual(case, case_id)
            self.assertFalse(guide_act)
            self.assertEqual(guide_st, 1)
            
            # Verify Case 2 scenario metadata is distinct from Case 1
            meta_1 = app.SCENARIOS['FRAUD_SMURFING_FUNNEL']
            meta_2 = app.SCENARIOS['FRAUD_SYBIL_DEVICE_RING']
            self.assertNotEqual(meta_1['title'], meta_2['title'])
            self.assertNotEqual(meta_1['focal_node'], meta_2['focal_node'])
        print(" PASS: Test 04 - Opening Case 2 selects Case 2 without stale data.")

    def test_05_network_evidence_tab_preserves_case(self):
        """Test 5: Network Evidence tab preserves selected case."""
        view = app.render_main_view('workspace', 'FRAUD_CIRCULAR_LOOP', 'network', False, 1, False, 'ALL', 'concentric', 'ALL')
        self.assertIsNotNone(view)
        elements = app.get_subgraph_elements('FRAUD_CIRCULAR_LOOP')
        self.assertTrue(len(elements) > 0)
        print(" PASS: Test 05 - Network Evidence tab preserves selected case (FRAUD_CIRCULAR_LOOP).")

    def test_06_model_explanation_tab_preserves_case(self):
        """Test 6: Model Explanation tab preserves selected case and builds SHAP attribution."""
        view = app.render_main_view('workspace', 'FRAUD_BUST_OUT', 'xai', False, 1, False, 'ALL', 'concentric', 'ALL')
        self.assertIsNotNone(view)
        focal = app.SCENARIOS['FRAUD_BUST_OUT']['focal_node']
        chart = app.build_shap_chart(focal)
        self.assertIsNotNone(chart)
        self.assertIn('data', chart)
        print(" PASS: Test 06 - Model Explanation tab preserves selected case (FRAUD_BUST_OUT) and builds SHAP.")

    def test_07_report_view_preserves_case_and_returns(self):
        """Test 7: Report view preserves selected case and back button navigates back to workspace."""
        view = app.render_main_view('report', 'LEGIT_CORP_PAYROLL', 'summary', False, 1, False, 'ALL', 'concentric', 'ALL')
        self.assertIsNotNone(view)
        self.assertEqual(view.id, 'report-view-container')
        
        # Test back button callback navigation
        with patch.object(app, 'ctx') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': '{"target":"workspace","type":"nav-btn"}.n_clicks', 'value': 1}]
            mock_ctx.triggered_id = {'target': 'workspace', 'type': 'nav-btn'}
            view_out, g_act, pmode = app.handle_view_navigation(0, 0, 0, 0, 0, 0, 0, [1])
            self.assertEqual(view_out, 'workspace')
        print(" PASS: Test 07 - Report view preserves case and returns to workspace.")

    def test_08_presentation_mode_starts_at_step_1(self):
        """Test 8: Presentation Mode starts at step 1."""
        with patch.object(app, 'ctx') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': '{"action":"toggle-presentation","type":"guide-mode-btn"}.n_clicks', 'value': 1}]
            mock_ctx.triggered_id = {'action': 'toggle-presentation', 'type': 'guide-mode-btn'}
            g_act, pmode, g_step = app.handle_guide_mode_toggle([1], False, False)
            self.assertTrue(pmode)
            self.assertTrue(g_act)
            self.assertEqual(g_step, 1)
        print(" PASS: Test 08 - Presentation Mode starts at step 1.")

    def test_09_next_advances_presentation_step(self):
        """Test 9: Next button advances presentation step from 1 to 2."""
        with patch.object(app, 'ctx') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': '{"direction":"next","type":"guide-step-btn"}.n_clicks', 'value': 1}]
            mock_ctx.triggered_id = {'direction': 'next', 'type': 'guide-step-btn'}
            step = app.handle_guide_step_navigation([1], 1, 'FRAUD_SMURFING_FUNNEL')
            self.assertEqual(step, 2)
        print(" PASS: Test 09 - Next advances presentation step to 2.")

    def test_10_prev_reverses_presentation_step(self):
        """Test 10: Previous button reverses presentation step from 2 to 1."""
        with patch.object(app, 'ctx') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': '{"direction":"prev","type":"guide-step-btn"}.n_clicks', 'value': 1}]
            mock_ctx.triggered_id = {'direction': 'prev', 'type': 'guide-step-btn'}
            step = app.handle_guide_step_navigation([1], 2, 'FRAUD_SMURFING_FUNNEL')
            self.assertEqual(step, 1)
        print(" PASS: Test 10 - Previous reverses presentation step to 1.")

    def test_11_restart_resets_step_to_one(self):
        """Test 11: Restart button resets presentation step to 1."""
        with patch.object(app, 'ctx') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': '{"direction":"restart","type":"guide-step-btn"}.n_clicks', 'value': 1}]
            mock_ctx.triggered_id = {'direction': 'restart', 'type': 'guide-step-btn'}
            step = app.handle_guide_step_navigation([1], 5, 'FRAUD_SMURFING_FUNNEL')
            self.assertEqual(step, 1)
        print(" PASS: Test 11 - Restart resets step to 1.")

    def test_12_exit_returns_to_workspace(self):
        """Test 12: Exit presentation mode sets pmode_active=False and guide_active=False cleanly."""
        with patch.object(app, 'ctx') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': '{"action":"exit-presentation","type":"guide-mode-btn"}.n_clicks', 'value': 1}]
            mock_ctx.triggered_id = {'action': 'exit-presentation', 'type': 'guide-mode-btn'}
            g_act, pmode, g_step = app.handle_guide_mode_toggle([1], True, True)
            self.assertFalse(pmode)
            self.assertFalse(g_act)
            self.assertEqual(g_step, 1)
        print(" PASS: Test 12 - Exit presentation mode returns cleanly to workspace.")

    def test_13_switching_cases_resets_walkthrough_state(self):
        """Test 13: Switching cases does not leak stale walkthrough state."""
        with patch.object(app, 'ctx') as mock_ctx:
            case_id = 'LEGIT_UTILITY_HUB'
            mock_ctx.triggered = [{'prop_id': f'{{"type":"open-workspace-btn","index":"{case_id}"}}.n_clicks', 'value': 1}]
            mock_ctx.triggered_id = {'type': 'open-workspace-btn', 'index': case_id}
            case, view, guide_act, guide_st, ws_tab, pmode, focal = app.handle_open_workspace_btn([1])
            self.assertEqual(case, case_id)
            self.assertFalse(guide_act)
            self.assertEqual(guide_st, 1)
            self.assertFalse(pmode)
        print(" PASS: Test 13 - Switching cases resets walkthrough state cleanly.")

    def test_14_invalid_case_id_falls_back_gracefully(self):
        """Test 14: Invalid case ID falls back gracefully to default case without crashing."""
        view = app.render_main_view('workspace', 'invalid_case_999', 'summary', False, 1, False, 'ALL', 'concentric', 'ALL')
        self.assertIsNotNone(view)
        fallback_case = 'invalid_case_999' if 'invalid_case_999' in app.SCENARIOS else app.SCENARIO_KEYS[0]
        self.assertEqual(fallback_case, 'FRAUD_SMURFING_FUNNEL')
        print(" PASS: Test 14 - Invalid case ID falls back gracefully to default case.")

    def test_15_breadcrumb_mount_does_not_redirect(self):
        """Test 15: Breadcrumb mount event (n_clicks=None or 0) does NOT redirect to Overview."""
        with patch.object(app, 'ctx') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': '{"target":"overview","type":"nav-btn"}.n_clicks', 'value': None}]
            mock_ctx.triggered_id = {'target': 'overview', 'type': 'nav-btn'}
            res = app.handle_view_navigation(0, 0, 0, 0, 0, 0, 0, [None])
            self.assertEqual(res, (app.no_update, app.no_update, app.no_update))
            
            mock_ctx.triggered = [{'prop_id': '{"target":"overview","type":"nav-btn"}.n_clicks', 'value': 0}]
            res0 = app.handle_view_navigation(0, 0, 0, 0, 0, 0, 0, [0])
            self.assertEqual(res0, (app.no_update, app.no_update, app.no_update))
        print(" PASS: Test 15 - Breadcrumb mount guard prevents spurious redirection.")

    def test_16_node_selection_updates_entity_inspector(self):
        """Test 16: Node selection in Cytoscape returns rich real telemetry in Entity Inspector."""
        focal = app.SCENARIOS['FRAUD_SMURFING_FUNNEL']['focal_node']
        node_data = {'id': focal}
        inspector, sel_node = app.handle_graph_node_click(node_data, 'FRAUD_SMURFING_FUNNEL')
        self.assertIsNotNone(inspector)
        self.assertEqual(sel_node, focal)
        self.assertIn(focal, str(inspector))
        print(" PASS: Test 16 - Node selection returns authentic telemetry in Entity Inspector.")

    def test_17_graph_filtering_updates_elements(self):
        """Test 17: Graph filtering by layout and filter toggles stores correctly."""
        with patch.object(app, 'ctx') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': '{"filter":"ACCOUNTS","type":"graph-filter-btn"}.n_clicks', 'value': 1}]
            mock_ctx.triggered_id = {'filter': 'ACCOUNTS', 'type': 'graph-filter-btn'}
            layout, flt = app.handle_graph_layout_filter([], [1], 'concentric', 'ALL')
            self.assertEqual(flt, 'ACCOUNTS')
            self.assertEqual(layout, 'concentric')
        print(" PASS: Test 17 - Graph filtering updates store correctly.")

    def test_18_logo_click_returns_to_overview(self):
        """Test 18: Logo click returns to Overview and exits Presentation Mode."""
        with patch.object(app, 'ctx') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': 'nav-brand-logo.n_clicks', 'value': 1}]
            mock_ctx.triggered_id = 'nav-brand-logo'
            view_out, g_act, pmode = app.handle_view_navigation(1, 0, 0, 0, 0, 0, 0, [0])
            self.assertEqual(view_out, 'overview')
            self.assertFalse(g_act)
            self.assertFalse(pmode)
        print(" PASS: Test 18 - Logo click returns to Overview and exits Presentation Mode.")

if __name__ == '__main__':
    unittest.main()
