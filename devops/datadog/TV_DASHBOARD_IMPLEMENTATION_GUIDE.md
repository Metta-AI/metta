# TV Dashboard Implementation Guide

**Goal**: Display the Datadog infrastructure health dashboard on a company TV screen, as envisioned by Nishad.

---

## 📚 Learning Resources (How to Learn This Yourself)

### 1. **Datadog Fundamentals**
- **Official Docs**: Start with [Datadog's Getting Started Guide](https://docs.datadoghq.com/getting_started/)
- **Metrics API**: [Datadog Metrics API v2 Documentation](https://docs.datadoghq.com/api/latest/metrics/)
- **Dashboards**: [Building Dashboards Guide](https://docs.datadoghq.com/dashboards/)
- **TV Mode**: [Datadog Screenboards & TV Mode](https://docs.datadoghq.com/dashboards/screenboard/)
- **Practice**: Create a free Datadog trial account and build a test dashboard

### 2. **Kubernetes & Helm**
- **Kubernetes Basics**: [Kubernetes.io Interactive Tutorial](https://kubernetes.io/docs/tutorials/kubernetes-basics/)
- **Helm**: [Helm Documentation](https://helm.sh/docs/)
- **CronJobs**: [Kubernetes CronJob Guide](https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/)
- **Practice**: Use `minikube` or `kind` locally to test deployments

### 3. **Python & APIs**
- **Datadog Python Client**: [datadog-api-client-python](https://github.com/DataDog/datadog-api-client-python)
- **REST APIs**: General understanding of HTTP requests, JSON payloads
- **Practice**: Write small scripts that call the Datadog API

### 4. **TV Display / Kiosk Mode**
- **Chrome Kiosk Mode**: [Chrome Kiosk Mode Guide](https://support.google.com/chrome/a/answer/9530001)
- **Raspberry Pi / TV Setup**: Common patterns for office displays
- **Auto-refresh**: Browser extensions or JavaScript for auto-refresh

### 5. **Learning Strategy**
1. **Start Small**: Build one collector locally, test it, then deploy
2. **Read Error Messages**: Kubernetes/Helm errors are usually descriptive
3. **Use `--dry-run`**: Test everything locally before pushing to production
4. **Ask Questions**: The infra team (Nishad) can clarify requirements
5. **Iterate**: Start with a simple dashboard, then add complexity

---

## 🖥️ TV Dashboard Implementation Approach

### Phase 1: Build the Dashboard in Datadog UI

**Steps:**
1. **Verify Metrics Are Flowing**
   - Go to Datadog → Metrics Explorer
   - Query: `metta.infra.cron.ci.workflow.success`
   - Confirm you see data points

2. **Create a Screenboard** (not a Timeboard)
   - Screenboards are better for TV displays (fixed layout, no scrolling)
   - Go to: Dashboards → New Dashboard → New Screenboard

3. **Build the Heatmap View** (per Nishad's mockup)
   - Use **Query Table** widgets or **Change** widgets
   - Query pattern:
     ```
     avg:last_15m:metta.infra.cron.ci.workflow.success{workflow_name:arena_multi_gpu,status:pass}
     ```
   - Conditional formatting:
     - Green when `value == 1` (pass)
     - Red when `value == 0` (fail)
     - Yellow when `value == null` (unknown)

4. **Add Composite Tiles**
   - CI smoothness: Use Distribution metrics for `p90(workflow_duration)`
   - Bug counts: `sum:metta.infra.cron.github.bugs.count{label:Training}`
   - PR throughput: `sum:metta.infra.cron.github.reverts.count`

5. **Layout for TV**
   - Use large fonts (minimum 18px)
   - High contrast colors
   - Group related metrics together
   - Leave margins (TVs may crop edges)

### Phase 2: Enable TV/Presentation Mode

**Option A: Datadog Native TV Mode** (Recommended)
1. In your Screenboard, click **"..."** menu → **"TV Mode"**
2. Datadog generates a TV-optimized URL
3. This URL:
   - Auto-refreshes (configurable interval)
   - Hides navigation/UI chrome
   - Full-screen friendly
   - Can be password-protected

**Option B: Custom Kiosk Setup**
If you need more control:

1. **Get the Dashboard URL**
   - Share → Get Shareable Link
   - Add `?tv=true` parameter if available

2. **Set Up a Kiosk Device** (Raspberry Pi, Chromebox, or dedicated machine)
   ```bash
   # On Linux/Raspberry Pi:
   # Install Chrome/Chromium
   # Create kiosk script:
   ```
   ```bash
   #!/bin/bash
   xset s off          # Disable screen saver
   xset -dpms          # Disable power management
   xset s noblank      # Disable blanking

   # Launch Chrome in kiosk mode
   chromium-browser \
     --kiosk \
     --autoplay-policy=no-user-gesture-required \
     --disable-infobars \
     --noerrdialogs \
     --disable-session-crashed-bubble \
     "https://app.datadoghq.com/screen/your-dashboard-id?tv=true"
   ```

3. **Auto-refresh** (if Datadog TV mode doesn't auto-refresh)
   - Use a browser extension like "Auto Refresh Plus"
   - Or inject JavaScript:
   ```javascript
   // Add to dashboard via browser console or extension
   setInterval(() => location.reload(), 300000); // 5 minutes
   ```

### Phase 3: Hardware Setup

**TV Display Options:**

1. **Dedicated Device** (Best for reliability)
   - Raspberry Pi 4 or Chromebox
   - Connect to TV via HDMI
   - Auto-boot into kiosk mode
   - Set up auto-reconnect on network issues

2. **Laptop/Computer** (Quick setup)
   - Dedicated laptop connected to TV
   - Set Chrome to open dashboard on startup
   - Disable sleep/screen saver

3. **Smart TV Browser** (If TV supports it)
   - Some smart TVs have browsers
   - Less reliable, but no extra hardware

**Network Considerations:**
- Ensure TV device has stable internet
- Consider wired Ethernet for reliability
- Set up monitoring for the kiosk device itself

---

## 🏗️ Proper Implementation Architecture

### Current State (What You Have)
```
K8s CronJob (every 10 min)
  → Python collectors (devops/datadog/collectors/)
  → Datadog Metrics API
  → Metrics stored in Datadog
```

### What's Missing
```
Datadog Dashboard (UI)
  → TV Mode URL
  → Kiosk Device
  → TV Screen
```

### Implementation Checklist

**Step 1: Verify Data Pipeline** ✅ (You're working on this)
- [ ] Metrics are being emitted correctly
- [ ] Tags are properly formatted
- [ ] Data appears in Datadog Metrics Explorer

**Step 2: Build Dashboard** (Next)
- [ ] Create Screenboard in Datadog UI
- [ ] Add heatmap widgets (Query Table with conditional formatting)
- [ ] Add composite tiles (CI smoothness, bug counts, etc.)
- [ ] Test dashboard locally in browser

**Step 3: Enable TV Mode**
- [ ] Enable Datadog TV Mode on dashboard
- [ ] Test TV mode URL in browser
- [ ] Verify auto-refresh works

**Step 4: Set Up Kiosk Device**
- [ ] Choose hardware (Raspberry Pi / Chromebox / laptop)
- [ ] Install OS and browser
- [ ] Configure kiosk mode script
- [ ] Test auto-boot and reconnection

**Step 5: Deploy to TV**
- [ ] Connect device to TV
- [ ] Configure TV input/settings
- [ ] Test full-screen display
- [ ] Document TV location and access

**Step 6: Monitoring & Maintenance**
- [ ] Set up alerts if kiosk device goes offline
- [ ] Document how to update dashboard
- [ ] Create runbook for common issues

---

## 🎯 Best Practices for TV Dashboards

### Design Principles
1. **Large, Readable Text**: Minimum 18px font, high contrast
2. **Color Coding**:
   - Green = Healthy/Pass
   - Red = Unhealthy/Fail
   - Yellow = Warning/Unknown
3. **Minimal Information**: Show only critical metrics
4. **No Scrolling**: Everything should fit on one screen
5. **Auto-refresh**: 5-15 minute intervals (don't refresh too often)

### Technical Best Practices
1. **Use Screenboards**: Better for fixed layouts than Timeboards
2. **Cache-Friendly**: Datadog TV mode handles caching well
3. **Error Handling**: Dashboard should show "No data" gracefully
4. **Network Resilience**: Kiosk should auto-reconnect on network issues
5. **Backup Plan**: Have a way to manually refresh if auto-refresh fails

### Security Considerations
1. **Read-Only Access**: TV dashboard should use read-only Datadog API keys
2. **Password Protection**: Enable password on TV mode if sensitive data
3. **Network Isolation**: Consider putting kiosk on isolated network segment
4. **Access Control**: Limit who can modify the dashboard

---

## 🐛 Common Issues & Solutions

### Issue: Dashboard shows "No data"
- **Check**: Are metrics being emitted? (Metrics Explorer)
- **Check**: Are tags matching? (Query tags must match emitted tags)
- **Check**: Time range (use `last_15m` or `last_1h`)

### Issue: TV mode doesn't auto-refresh
- **Solution**: Use browser extension or JavaScript injection
- **Solution**: Use Datadog's native TV mode (if available)

### Issue: Kiosk device loses connection
- **Solution**: Set up network monitoring
- **Solution**: Use wired Ethernet instead of WiFi
- **Solution**: Add auto-reconnect script

### Issue: Dashboard looks bad on TV
- **Solution**: Adjust TV resolution/zoom settings
- **Solution**: Use Datadog's TV-optimized layout
- **Solution**: Test on actual TV before final deployment

---

## 📋 Next Steps for Your Implementation

1. **Finish the Data Pipeline** (current work)
   - Complete collectors
   - Verify metrics in Datadog

2. **Build Dashboard Prototype**
   - Create a simple Screenboard with 2-3 widgets
   - Test locally
   - Get feedback from Nishad

3. **Set Up Test TV Display**
   - Use a spare monitor/laptop first
   - Test kiosk mode
   - Refine dashboard layout

4. **Deploy to Production TV**
   - Set up dedicated kiosk device
   - Connect to office TV
   - Document setup

---

## 🔗 Useful Links

- [Datadog Screenboards](https://docs.datadoghq.com/dashboards/screenboard/)
- [Datadog Metrics API](https://docs.datadoghq.com/api/latest/metrics/)
- [Chrome Kiosk Mode](https://support.google.com/chrome/a/answer/9530001)
- [Raspberry Pi Kiosk Setup](https://github.com/futurice/chilipie-kiosk)
- [Kubernetes CronJobs](https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/)

---

## 💡 Pro Tips

1. **Start Simple**: Build a basic dashboard first, then add complexity
2. **Test Locally**: Always test in browser before deploying to TV
3. **Get Feedback**: Show Nishad early iterations for alignment
4. **Document Everything**: Future you (and others) will thank you
5. **Monitor the Monitor**: Set up alerts for the kiosk device itself

---

**Remember**: You're not expected to know everything upfront. This is a learning opportunity. Start small, iterate, and ask questions when stuck. The infra team is there to help!

