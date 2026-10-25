import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import unittest
from unittest.mock import patch
import app

class FraudCircleAdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n" + "="*60)
        print(" RUNNING FRAUDCIRCLE ADVERSARIAL QA & STRESS TEST SUITE")
        print("="*60 + "\n")

    def test_01_view_containers_have_explicit_ids(self):
        """Test 1: Every top-level view contains a distinctive root container ID."""
        views_to_test = [
            ('overview', 'overview-view-container'),
            ('cases', 'cases-view-container'),
            ('workspace', 'workspace-view-container'),
            ('report', 'report-view-container'),
            ('benchmarks', 'benchmarks-view-container'),
            ('about', 'about-view-container')
        ]
        for v_name, expected_id in views_to_test:
            rendered = app.render_main_view(v_name, 'FRAUD_SMURFING_FUNNEL', 'summary', False, 1, False, 'ALL', 'concentric', 'ALL')
            self.assertEqual(rendered.id, expected_id, f"View {v_name} missing root id {expected_id}")
        print(" PASS: Test 01 - All 6 views contain explicit root container IDs.")

    def test_02_all_7_cases_open_cleanly_with_focal_node(self):
        """Test 2: Opening each of the 7 cases resets presentation/guide and pre-selects focal node."""
        for case_key in app.SCENARIO_KEYS:
            with patch.object(app, 'ctx') as mock_ctx:
                mock_ctx.triggered = [{'prop_id': f'{{"index":"{case_key}","type":"open-workspace-btn"}}.n_clicks', 'value': 1}]
                mock_ctx.triggered_id = {'index': case_key, 'type': 'open-workspace-btn'}
                c_out, v_out, g_act, g_step, tab, pmode, focal = app.handle_open_workspace_btn([1])
                self.assertEqual(c_out, case_key)
                self.assertEqual(v_out, 'workspace')
                self.assertFalse(g_act)
                self.assertEqual(g_step, 1)
                self.assertEqual(tab, 'summary')
                self.assertFalse(pmode)
                self.assertEqual(focal, app.SCENARIOS[case_key]['focal_node'])
        print(" PASS: Test 02 - All 7 cases open cleanly with proper focal node initialization.")

    def test_03_presentation_mode_finish_button_exits_smoothly(self):
        """Test 3: At Step 6, 'Finish Walkthrough' triggers exit-presentation and returns (False, False, 1)."""
        sc = app.SCENARIOS['FRAUD_SMURFING_FUNNEL']
        max_steps = len(sc['walkthrough'])
        self.assertEqual(max_steps, 6)

        # In render_workspace_view at step 6, button has action: exit-presentation
        rendered_ws = app.render_workspace_view('FRAUD_SMURFING_FUNNEL', 'network', True, 6, True, 'concentric', 'ALL')
        self.assertIsNotNone(rendered_ws)

        # Now simulate clicking 'Finish Walkthrough' (action="exit-presentation")
        with patch.object(app, 'ctx') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': '{"action":"exit-presentation","type":"guide-mode-btn"}.n_clicks', 'value': 1}]
            mock_ctx.triggered_id = {'action': 'exit-presentation', 'type': 'guide-mode-btn'}
            g_act, pmode, g_step = app.handle_guide_mode_toggle([1], True, True)
            self.assertFalse(g_act, "guide_active must be False after finish")
            self.assertFalse(pmode, "presentation_mode must be False after finish")
            self.assertEqual(g_step, 1, "guide_step must reset to 1 after finish")
        print(" PASS: Test 03 - Finish Walkthrough button exits presentation mode smoothly.")

    def test_04_presentation_mode_exit_button_resets_cleanly(self):
        """Test 4: Clicking 'Exit Presentation' returns (False, False, 1) to restore normal workspace."""
        with patch.object(app, 'ctx') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': '{"action":"exit-presentation","type":"guide-mode-btn"}.n_clicks', 'value': 1}]
            mock_ctx.triggered_id = {'action': 'exit-presentation', 'type': 'guide-mode-btn'}
            g_act, pmode, g_step = app.handle_guide_mode_toggle([1], True, True)
            self.assertFalse(pmode)
            self.assertFalse(g_act)
            self.assertEqual(g_step, 1)
        print(" PASS: Test 04 - Exit Presentation resets to clean workspace at step 1.")

    def test_05_navigating_away_from_workspace_kills_presentation_mode(self):
        """Test 5: Leaving workspace via navbar/breadcrumbs resets presentation_mode and guide_active."""
        destinations = ['overview', 'cases', 'benchmarks', 'report', 'about']
        for dest in destinations:
            with patch.object(app, 'ctx') as mock_ctx:
                mock_ctx.triggered = [{'prop_id': f'nav-btn-{dest}.n_clicks', 'value': 1}]
                mock_ctx.triggered_id = f'nav-btn-{dest}'
                v_out, g_act, pmode = app.handle_view_navigation(0, 0, 0, 0, 0, 0, 0, [0])
                self.assertEqual(v_out, dest)
                self.assertFalse(g_act, f"guide_active leaked when navigating to {dest}")
                self.assertFalse(pmode, f"presentation_mode leaked when navigating to {dest}")
        print(" PASS: Test 05 - Navigating away from workspace unconditionally kills presentation mode.")

    def test_06_two_way_url_store_synchronization(self):
        """Test 6: Store changes update URL pathname/search, and URL back/forward updates Store."""
        # 1. Store -> URL: Workspace Case 3
        path, search = app.sync_store_to_url('workspace', 'FRAUD_CIRCULAR_LOOP', '/', '')
        self.assertEqual(path, '/workspace')
        self.assertEqual(search, '?case=FRAUD_CIRCULAR_LOOP')

        # 2. Store -> URL: Overview has no search query
        path_ov, search_ov = app.sync_store_to_url('overview', 'FRAUD_CIRCULAR_LOOP', '/workspace', '?case=FRAUD_CIRCULAR_LOOP')
        self.assertEqual(path_ov, '/')
        self.assertEqual(search_ov, '')

        # 3. URL -> Store: Browser back to /cases
        v_out, c_out = app.sync_url_to_store('/cases', '', 'workspace', 'FRAUD_CIRCULAR_LOOP')
        self.assertEqual(v_out, 'cases')
        self.assertEqual(c_out, 'FRAUD_CIRCULAR_LOOP')

        # 4. URL -> Store: Direct link to /workspace?case=FRAUD_BUST_OUT
        v_direct, c_direct = app.sync_url_to_store('/workspace', '?case=FRAUD_BUST_OUT', 'overview', 'FRAUD_SMURFING_FUNNEL')
        self.assertEqual(v_direct, 'workspace')
        self.assertEqual(c_direct, 'FRAUD_BUST_OUT')
        print(" PASS: Test 06 - Two-way URL synchronization and deep linking validated.")

    def test_07_audit_trail_persistence_and_case_isolation(self):
        """Test 7: HITL actions persist in audit-log-store per case without cross-contamination."""
        # 1. Quarantine focal node on Case 1
        with patch.object(app, 'ctx') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': '{"action":"quarantine","type":"hitl-action-btn"}.n_clicks', 'value': 1}]
            mock_ctx.triggered_id = {'action': 'quarantine', 'type': 'hitl-action-btn'}
            log_dict = app.handle_hitl_action([1], 'FRAUD_SMURFING_FUNNEL', {})
            self.assertIn('FRAUD_SMURFING_FUNNEL', log_dict)
            self.assertEqual(len(log_dict['FRAUD_SMURFING_FUNNEL']), 1)
            self.assertEqual(log_dict['FRAUD_SMURFING_FUNNEL'][0]['badge'], 'QUARANTINE ORDER')

        # 2. Whitelist focal node on Case 5 (Infosys Payroll)
        with patch.object(app, 'ctx') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': '{"action":"whitelist","type":"hitl-action-btn"}.n_clicks', 'value': 1}]
            mock_ctx.triggered_id = {'action': 'whitelist', 'type': 'hitl-action-btn'}
            log_dict = app.handle_hitl_action([1], 'LEGIT_CORP_PAYROLL', log_dict)
            self.assertIn('LEGIT_CORP_PAYROLL', log_dict)
            self.assertEqual(len(log_dict['LEGIT_CORP_PAYROLL']), 1)
            self.assertEqual(log_dict['LEGIT_CORP_PAYROLL'][0]['badge'], 'EXEMPTION ISSUED')

        # Verify Case 1 and Case 5 remain completely isolated
        self.assertNotEqual(log_dict['FRAUD_SMURFING_FUNNEL'][0]['badge'], log_dict['LEGIT_CORP_PAYROLL'][0]['badge'])

        # 3. Verify workspace view renders persisted entries
        ws_view = app.render_workspace_view('FRAUD_SMURFING_FUNNEL', 'report', False, 1, False, 'concentric', 'ALL', audit_logs=log_dict)
        self.assertIn("QUARANTINE ORDER", str(ws_view))
        self.assertNotIn("EXEMPTION ISSUED", str(ws_view))
        print(" PASS: Test 07 - Audit trail persists across rerenders with strict per-case isolation.")

    def test_08_entity_inspector_pre_populates_and_persists_selection(self):
        """Test 8: Node clicks update entity inspector and selected-node-store persists across rerenders."""
        focal = app.SCENARIOS['FRAUD_SMURFING_FUNNEL']['focal_node']
        # Node click
        with patch.object(app, 'ctx') as mock_ctx:
            inspector, sel_node = app.handle_graph_node_click({'id': focal}, 'FRAUD_SMURFING_FUNNEL')
            self.assertEqual(sel_node, focal)
            self.assertIn(focal, str(inspector))

        # Rerendering workspace view with selected_node pre-populates inspector
        ws_rendered = app.render_workspace_view('FRAUD_SMURFING_FUNNEL', 'network', False, 1, False, 'concentric', 'ALL', selected_node=focal)
        self.assertIn(focal, str(ws_rendered))
        print(" PASS: Test 08 - Entity inspector pre-populates and persists selected node.")

    def test_09_all_cases_filter_toggles(self):
        """Test 9: Cases view filters ALL, FRAUD, and LEGIT correctly."""
        for flt in ['ALL', 'FRAUD', 'LEGIT']:
            with patch.object(app, 'ctx') as mock_ctx:
                mock_ctx.triggered = [{'prop_id': f'{{"filter":"{flt}","type":"cases-filter-btn"}}.n_clicks', 'value': 1}]
                mock_ctx.triggered_id = {'filter': flt, 'type': 'cases-filter-btn'}
                res = app.handle_cases_filter([1])
                self.assertEqual(res, flt)
                view = app.render_cases_view(filter_val=flt)
                self.assertIsNotNone(view)
        print(" PASS: Test 09 - Cases view filters ALL, FRAUD, and LEGIT correctly.")

    def test_10_modals_open_and_close_without_error(self):
        """Test 10: FIU and Neo4j modals open and close cleanly."""
        # FIU Open & Close
        with patch.object(app, 'ctx') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': '{"modal":"fiu","type":"open-modal-btn"}.n_clicks', 'value': 1}]
            mock_ctx.triggered_id = {'modal': 'fiu', 'type': 'open-modal-btn'}
            is_open = app.handle_fiu_modal(1, 0, False)
            self.assertTrue(is_open)

            mock_ctx.triggered = [{'prop_id': 'btn-close-fiu-modal.n_clicks', 'value': 1}]
            mock_ctx.triggered_id = 'btn-close-fiu-modal'
            is_open_closed = app.handle_fiu_modal(0, 1, True)
            self.assertFalse(is_open_closed)

        # Neo4j Open & Close
        with patch.object(app, 'ctx') as mock_ctx:
            mock_ctx.triggered = [{'prop_id': '{"modal":"neo4j","type":"open-modal-btn"}.n_clicks', 'value': 1}]
            mock_ctx.triggered_id = {'modal': 'neo4j', 'type': 'open-modal-btn'}
            is_neo_open = app.handle_neo4j_modal(1, 0, False)
            self.assertTrue(is_neo_open)

            mock_ctx.triggered = [{'prop_id': 'btn-close-neo4j-modal.n_clicks', 'value': 1}]
            mock_ctx.triggered_id = 'btn-close-neo4j-modal'
            is_neo_closed = app.handle_neo4j_modal(0, 1, True)
            self.assertFalse(is_neo_closed)
        print(" PASS: Test 10 - FIU and Neo4j modals open and close cleanly.")

if __name__ == '__main__':
    unittest.main()
