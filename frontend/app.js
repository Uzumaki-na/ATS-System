/* ============================================
   The Lab — TriadRank ATS Showcase
   Application Logic
   ============================================ */

// ============================================
// Configuration
// ============================================
const CATEGORIES = [
  'ACCOUNTANT','ADVOCATE','AGRICULTURE','APPAREL','ARTS','AUTOMOBILE',
  'AVIATION','BANKING','BPO','BUSINESS-DEVELOPMENT','CHEF','CONSTRUCTION',
  'CONSULTANT','DESIGNER','DIGITAL-MEDIA','ENGINEERING','FINANCE','FITNESS',
  'HEALTHCARE','HUMAN-RESOURCES','INFORMATION-TECHNOLOGY','PUBLIC-RELATIONS',
  'SALES','TEACHER'
];

const CONFIG = {
  useMockData: true,        // false → calls real FastAPI backend
  apiBaseUrl: 'http://127.0.0.1:8000',
  animationDelay: 400,       // ms between terminal lines
  minProcessingTime: 2000,   // ms minimum for terminal animation
  categories: CATEGORIES,
};

const MOCK_CANDIDATES = generateMockCandidates(12);

// ============================================
// Mock Data Generator
// ============================================
function generateMockCandidates(count) {
  const firstNames = ['Alice','Bob','Carol','David','Eve','Frank','Grace','Henry','Iris','Jack','Kate','Liam'];
  const lastNames  = ['Chen','Smith','Patel','Jones','Lee','Kim','Brown','Davis','Wilson','Taylor','Anderson','Thomas'];
  const labels    = ['Good Fit','Potential Fit','Bad Fit'];
  const labelsMap = { 'Good Fit': 2, 'Potential Fit': 1, 'Bad Fit': 0 };

  const skillPool = [
    'Python','PyTorch','TensorFlow','scikit-learn','spaCy','BERT','NLP',
    'Data Analysis','SQL','AWS','Docker','Kubernetes','FastAPI','React',
    'TypeScript','Go','Rust','PostgreSQL','MongoDB','Redis','Kafka',
    'Tableau','Power BI','Excel','Java','C++','Ruby','Node.js','GraphQL',
    'Machine Learning','Deep Learning','Computer Vision','LLMs','RAG'
  ];

  return Array.from({ length: count }, (_, i) => {
    const sem    = +(0.5 + Math.random() * 0.5).toFixed(3);
    const kw     = +(0.3 + Math.random() * 0.6).toFixed(3);
    const exp    = +(0.4 + Math.random() * 0.5).toFixed(3);
    const edu    = +(0.6 + Math.random() * 0.4).toFixed(3);
    const keyw   = +(0.3 + Math.random() * 0.6).toFixed(3);
    const raw    = +(sem * 0.3 + kw * 0.25 + exp * 0.2 + edu * 0.15 + keyw * 0.1).toFixed(4);
    const penalty = Math.random() > 0.7 ? 0.5 : 1.0;
    const final  = +(raw * penalty).toFixed(4);
    const label  = final >= 0.6 ? 'Good Fit' : final >= 0.35 ? 'Potential Fit' : 'Bad Fit';
    const catMatch = Math.random() > 0.15;

    const skills = Array.from(
      { length: 3 + Math.floor(Math.random() * 6) },
      () => skillPool[Math.floor(Math.random() * skillPool.length)]
    ).filter((v, idx, a) => a.indexOf(v) === idx);

    return {
      id: `CAND-${String(i + 1).padStart(3, '0')}`,
      name: `${firstNames[i % firstNames.length]} ${lastNames[i % lastNames.length]}`,
      final_score: final,
      raw_score: raw,
      label: label,
      label_probabilities: {
        'Good Fit': label === 'Good Fit' ? 0.7 + Math.random() * 0.25 : 0.05 + Math.random() * 0.2,
        'Potential Fit': label === 'Potential Fit' ? 0.6 + Math.random() * 0.2 : 0.1 + Math.random() * 0.3,
        'Bad Fit': label === 'Bad Fit' ? 0.6 + Math.random() * 0.25 : 0.05 + Math.random() * 0.15,
      },
      category: {
        predicted: CATEGORIES[Math.floor(Math.random() * CATEGORIES.length)],
        match: catMatch,
        confidence: +(0.65 + Math.random() * 0.3).toFixed(3),
      },
      skill_overlap: +(Math.random() * 0.8).toFixed(3),
      keyword_overlap: +keyw,
      extracted_skills: skills,
      radar: { sem, kw, exp, edu, keyw },
      penalties: [
        catMatch ? null : { type: 'Category mismatch', value: -0.05 },
        raw < 0.4 ? { type: 'Low model confidence', value: -0.02 } : null,
        skills.length < 4 ? { type: 'Missing skills coverage', value: -0.03 } : null,
        Math.random() > 0.8 ? { type: 'Experience gap detected', value: -0.015 } : null,
      ].filter(Boolean),
    };
  });
}

// ============================================
// API Layer
// ============================================
const API = {
  async rankCandidates(jd, category, candidates) {
    if (CONFIG.useMockData) {
      return this._mockRank(jd, category, candidates);
    }
    return this._realRank(jd, category, candidates);
  },

  async _mockRank(_jd, _category, _candidates) {
    // Simulate network delay
    await new Promise(r => setTimeout(r, 300));
    const results = MOCK_CANDIDATES
      .map((c, i) => ({ ...c, rank: i + 1 }))
      .sort((a, b) => b.final_score - a.final_score)
      .map((c, i) => ({ ...c, rank: i + 1 }));

    return {
      job_id: 'mock-' + Date.now(),
      job_category: _category,
      total_candidates: results.length,
      returned_candidates: results.length,
      processing_time_seconds: 2.1,
      results,
    };
  },

  async _realRank(jd, category, candidates) {
    const body = {
      job_description: jd,
      job_category: category,
      candidates: candidates.map(c => ({
        id: c.id,
        text: c.text || 'Mock resume text for ' + c.id,
      })),
    };

    const resp = await fetch(`${CONFIG.apiBaseUrl}/rank`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || `API error: ${resp.status}`);
    }

    return resp.json();
  },

  async fetchCategories() {
    if (CONFIG.useMockData) {
      return CONFIG.categories;
    }
    const resp = await fetch(`${CONFIG.apiBaseUrl}/categories`);
    const data = await resp.json();
    return data.categories;
  },

  async healthCheck() {
    if (CONFIG.useMockData) return { status: 'healthy (mock)' };
    const resp = await fetch(`${CONFIG.apiBaseUrl}/health`);
    return resp.json();
  },
};

// ============================================
// SPA Router
// ============================================
const Router = {
  _currentPage: null,
  _navLinks: null,
  _pages: null,

  init() {
    this._navLinks = document.querySelectorAll('.nav-links a');
    this._pages = document.querySelectorAll('.page');

    // Handle nav clicks
    this._navLinks.forEach(link => {
      link.addEventListener('click', e => {
        e.preventDefault();
        const page = link.getAttribute('data-page');
        this.navigate(page);
      });
    });

    // Handle hash changes
    window.addEventListener('hashchange', () => this._handleHash());

    // Initial route
    this._handleHash();
  },

  _handleHash() {
    const hash = window.location.hash.slice(1) || 'abstract';
    this._showPage(hash);
  },

  navigate(page) {
    window.location.hash = page;
  },

  _showPage(pageId) {
    if (this._currentPage === pageId) return;
    if (!document.getElementById(`page-${pageId}`)) return;

    // Update nav
    this._navLinks.forEach(l => l.classList.remove('active'));
    const activeLink = document.querySelector(`.nav-links a[data-page="${pageId}"]`);
    if (activeLink) activeLink.classList.add('active');

    // Update pages
    this._pages.forEach(p => p.classList.remove('active'));
    document.getElementById(`page-${pageId}`).classList.add('active');

    this._currentPage = pageId;

    // Page-specific hooks
    if (pageId === 'results') Results.loadResults();
    if (pageId === 'methodology') Methodology.init();

    // Re-trigger hero stagger animations when returning to abstract
    if (pageId === 'abstract') {
      const heroEls = document.querySelectorAll('.abstract-hero [class*="abstract-"], .abstract-hero h1, .abstract-hero .mono, .abstract-hero-text, .abstract-diagram, .abstract-cta-row');
      heroEls.forEach(el => {
        el.style.animation = 'none';
        void el.offsetHeight;
        el.style.animation = '';
      });
    }

    // Notify external hooks (scroll reveal init, etc.)
    if (this._onPageChange) this._onPageChange();
  },

  _onPageChange: null, // hook set by init

  getCurrentPage() { return this._currentPage; },
};

// ============================================
// Terminal Animation
// ============================================
const Terminal = {
  _element: null,
  _dropzone: null,
  _statusBar: null,
  _executeBtn: null,
  _abort: false,

  init() {
    this._element = document.getElementById('terminal-output');
    this._dropzone = document.getElementById('dropzone');
    this._statusBar = document.getElementById('sandbox-status');
    this._executeBtn = document.getElementById('btn-execute');
  },

  async simulateProcessing() {
    if (!this._element) return;
    this._abort = false;
    this._element.innerHTML = '';
    this._element.classList.add('active');
    if (this._dropzone) this._dropzone.style.display = 'none';
    if (this._executeBtn) this._executeBtn.disabled = true;
    if (this._statusBar) this._statusBar.innerHTML = '<span class="status-dot"></span> SYSTEM STATUS: PROCESSING';

    const lines = [
      { text: '── TriadRank Pipeline v1.0 ──', cls: 'info cursor' },
      { text: '', cls: '' },
      { text: '[INIT] Loading spaCy entity extractor...', cls: 'info' },
      { text: '[INIT] Extractor loaded (en_core_web_sm)', cls: 'success' },
      { text: '[INIT] Loading category encoder (DistilBERT)...', cls: 'info' },
      { text: '[INIT] Category encoder loaded (24 classes)', cls: 'success' },
      { text: '[INIT] Loading cross-encoder (BERT-base)...', cls: 'info' },
      { text: '[WARN] Cross-encoder checkpoint: best_model.pt', cls: 'warn' },
      { text: '[INIT] Cross-encoder loaded (regression + 3-class head)', cls: 'success' },
      { text: '', cls: '' },
      { text: '[TIER 3] Extracting entities from candidates...', cls: 'info' },
      { text: '[TIER 3] Computing keyword overlap (Jaccard)...', cls: 'info' },
      { text: '[TIER 3] Computing skill overlap (custom rules)...', cls: 'info' },
      { text: '[TIER 3] Selected top-K candidates', cls: 'success' },
      { text: '', cls: '' },
      { text: '[TIER 2] Validating resume categories...', cls: 'info' },
      { text: '[TIER 2] Applying category match penalties...', cls: 'info' },
      { text: '[TIER 2] Category validation complete', cls: 'success' },
      { text: '', cls: '' },
      { text: '[TIER 1] Running cross-encoder inference...', cls: 'info' },
      { text: '[TIER 1] Processing batch 1/1 (batch_size=16)', cls: 'info' },
      { text: '[TIER 1] Regression scores computed', cls: 'success' },
      { text: '[TIER 1] Classification labels assigned', cls: 'success' },
      { text: '', cls: '' },
      { text: '[RANK] Combining scores and sorting...', cls: 'info' },
      { text: '[RANK] Applying penalty adjustments...', cls: 'info' },
      { text: '', cls: '' },
      { text: '═══ Pipeline Complete ═══', cls: 'success cursor' },
      { text: `═══ ${MOCK_CANDIDATES.length} candidates ranked ═══`, cls: 'success' },
    ];

    for (let i = 0; i < lines.length; i++) {
      if (this._abort) break;

      const line = lines[i];
      const div = document.createElement('div');
      div.className = `terminal-line ${line.cls}`;
      div.textContent = line.text;
      div.style.animationDelay = `${i * CONFIG.animationDelay}ms`;
      this._element.appendChild(div);
      this._element.scrollTop = this._element.scrollHeight;

      // Wait for this line's animation
      await new Promise(r => setTimeout(r, CONFIG.animationDelay));
    }

    // Minimum processing feel
    await new Promise(r => setTimeout(r, 600));

    if (this._statusBar) this._statusBar.innerHTML = '<span class="status-dot"></span> SYSTEM STATUS: COMPLETE';
    if (this._executeBtn) this._executeBtn.disabled = false;

    // Navigate to results
    setTimeout(() => {
      Router.navigate('results');
    }, 500);
  },

  reset() {
    this._abort = true;
    if (this._element) {
      this._element.innerHTML = '';
      this._element.classList.remove('active');
    }
    if (this._dropzone) this._dropzone.style.display = 'flex';
    if (this._statusBar) this._statusBar.innerHTML = '<span class="status-dot"></span> SYSTEM STATUS: READY';
    if (this._executeBtn) this._executeBtn.disabled = false;
  },
};

// ============================================
// Input Sandbox — Drag & Drop
// ============================================
const Sandbox = {
  _files: [],
  _dropzone: null,
  _fileInput: null,

  init() {
    this._dropzone = document.getElementById('dropzone');
    this._fileInput = document.getElementById('file-input');
    this._setupDragDrop();
    this._setupFileInput();
  },

  _setupDragDrop() {
    if (!this._dropzone) return;

    this._dropzone.addEventListener('dragover', e => {
      e.preventDefault();
      this._dropzone.classList.add('drag-over');
    });

    this._dropzone.addEventListener('dragleave', () => {
      this._dropzone.classList.remove('drag-over');
    });

    this._dropzone.addEventListener('drop', e => {
      e.preventDefault();
      this._dropzone.classList.remove('drag-over');
      const files = Array.from(e.dataTransfer.files).filter(
        f => f.name.endsWith('.pdf') || f.name.endsWith('.txt')
      );
      if (files.length) this._addFiles(files);
    });

    this._dropzone.addEventListener('click', () => {
      if (!this._hasReachedMax()) {
        this._fileInput?.click();
      }
    });
  },

  _setupFileInput() {
    if (!this._fileInput) return;
    this._fileInput.addEventListener('change', e => {
      const files = Array.from(e.target.files);
      this._addFiles(files);
      e.target.value = '';
    });
  },

  _addFiles(files) {
    const maxTotal = 100;
    const available = maxTotal - this._files.length;
    const toAdd = files.slice(0, available);

    toAdd.forEach(f => {
      this._files.push({
        id: `file-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        name: f.name,
        size: f.size,
        file: f,
      });
    });

    this._renderFileTags();
  },

  _removeFile(id) {
    this._files = this._files.filter(f => f.id !== id);
    this._renderFileTags();
  },

  _renderFileTags() {
    if (!this._dropzone) return;

    if (this._files.length === 0) {
      this._dropzone.classList.remove('has-files');
      this._dropzone.innerHTML = `
        <div class="dropzone-icon">⤓</div>
        <div class="dropzone-text">
          Drop .PDF or .TXT files here<br>
          or click to browse
        </div>
      `;
      return;
    }

    this._dropzone.classList.add('has-files');
    // Simulate parsed/error states for visual richness
    this._dropzone.innerHTML = this._files.map((f, idx) => {
      const isCorrupted = idx === this._files.length - 1 && this._files.length > 2 && f.name.toLowerCase().includes('.txt');
      const isWarning = f.size > 500000;
      const statusClass = isCorrupted ? 'error' : isWarning ? 'warning' : 'parsed';
      const statusLabel = isCorrupted ? 'OCR_FAIL' : isWarning ? 'LARGE' : 'Parsed';
      const icon = isCorrupted ? 'warning' : 'description';
      const wrapperClass = isCorrupted ? 'file-token error' : isWarning ? 'file-token warning' : 'file-token';
      return `
        <div class="${wrapperClass}">
          <div class="file-token-left">
            <span class="file-icon material-symbols-outlined">${icon}</span>
            <span class="file-token-name">${this._truncateName(f.name, 30)}</span>
          </div>
          <div class="file-token-right">
            <span class="file-token-status ${statusClass}">| ${statusLabel}</span>
            <button class="file-token-remove" data-file-id="${f.id}" aria-label="Remove ${f.name}">
              <span class="material-symbols-outlined" style="font-size:16px">close</span>
            </button>
          </div>
        </div>
      `;
    }).join('');

    // Wire remove buttons
    this._dropzone.querySelectorAll('.file-token-remove').forEach(btn => {
      btn.addEventListener('click', e => {
        e.stopPropagation();
        this._removeFile(btn.getAttribute('data-file-id'));
      });
    });

    // Update the file input to reflect files
    const countEl = document.getElementById('file-count');
    if (countEl) countEl.textContent = `${this._files.length} file(s)`;
  },

  _truncateName(name, max) {
    if (name.length <= max) return name;
    const ext = name.lastIndexOf('.');
    const extStr = ext > 0 ? name.slice(ext) : '';
    const base = name.slice(0, max - extStr.length - 3);
    return base + '...' + extStr;
  },

  _hasReachedMax() {
    return this._files.length >= 100;
  },

  getFiles() { return this._files; },
  getFileCount() { return this._files.length; },
  clearFiles() {
    this._files = [];
    this._renderFileTags();
  },
};

// ============================================
// Results Matrix
// ============================================
const Results = {
  _data: [],
  _loading: false,

  async loadResults() {
    const resultsEl = document.getElementById('results-content');
    if (!resultsEl) return;

    this._loading = true;
    this._renderSkeleton(resultsEl);

    try {
      const jdEl = document.getElementById('jd-text');
      const jdText = jdEl ? jdEl.value || 'Software Engineer position requiring ML expertise' : '';
      const catEl = document.getElementById('category-select');
      const category = catEl ? catEl.value : 'ENGINEERING';

      const response = await API.rankCandidates(jdText, category, []);
      this._data = response.results;
      this._renderResults(resultsEl, response);
    } catch (err) {
      this._renderError(resultsEl, err);
    } finally {
      this._loading = false;
    }
  },

  _renderSkeleton(container) {
    container.innerHTML = Array.from({ length: 6 }, () => `
      <div class="skeleton-row">
        <div class="skeleton-cell"></div>
        <div class="skeleton-cell"></div>
        <div class="skeleton-cell"></div>
        <div class="skeleton-cell"></div>
        <div class="skeleton-cell"></div>
      </div>
    `).join('');
  },

  _renderResults(container, response) {
    const data = response.results;

    if (!data || data.length === 0) {
      container.innerHTML = `
        <div class="results-empty">
          <div class="results-empty-icon">&empty;</div>
          <div class="results-empty-text">
            No results yet. Run the sandbox to score candidates.
          </div>
          <button class="btn btn-secondary btn-sm" onclick="Router.navigate('sandbox')">
            &larr; Back to Sandbox
          </button>
        </div>
      `;
      return;
    }

    // Update toolbar count
    const countEl = document.getElementById('results-count');
    if (countEl) countEl.textContent = `${data.length} candidates ranked`;

    // Update sidebar params (use first result's data)
    this._updateSidebar(data[0]);

    // Render table
    container.innerHTML = `
      <table class="results-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Candidate</th>
            <th>Score</th>
            <th>Label</th>
            <th>Category</th>
            <th>Skills</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          ${data.map((c, i) => `
            <tr data-index="${i}" tabindex="0" class="${c.final_score < 0.4 ? 'row-disabled' : ''}">
              <td style="text-align:center;font-weight:600;background:var(--color-surface-container)">${String(i + 1).padStart(2, '0')}</td>
              <td>
                <div>${c.id || c.candidate_id}</div>
                <div style="font-size:11px;color:var(--color-muted)">${c.name || ''}</div>
              </td>
              <td class="score-cell">
                <span class="score-bar" style="width:${Math.round(c.final_score * 80)}px"></span>
                ${c.final_score.toFixed(4)}
              </td>
              <td>
                <span class="label-badge ${c.label === 'Good Fit' ? 'good-fit' : c.label === 'Potential Fit' ? 'potential-fit' : 'bad-fit'}">
                  ${c.label}
                </span>
              </td>
              <td>
                <span class="category-match ${c.category.match ? 'match' : 'mismatch'}">
                  ${c.category.match ? '&check;' : '&cross;'} ${c.category.predicted}
                </span>
              </td>
              <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
                ${c.extracted_skills?.slice(0, 3).join(', ') || '&mdash;'}
              </td>
              <td>
                <button class="table-action-btn ${c.final_score >= 0.6 ? '' : 'outline'}" data-index="${i}">
                  ${c.final_score >= 0.6 ? 'Review' : 'Discard'}
                </button>
              </td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;

    // Animate score bars after render
    setTimeout(animateScoreBars, 50);

    // Wire row clicks and action buttons
    container.querySelectorAll('tbody tr').forEach(row => {
      row.addEventListener('click', (e) => {
        if (e.target.closest('.table-action-btn')) return; // let action btn handle
        const idx = parseInt(row.getAttribute('data-index'), 10);
        this._openDetail(data[idx]);
      });
      row.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          const idx = parseInt(row.getAttribute('data-index'), 10);
          this._openDetail(data[idx]);
        }
      });
    });
    container.querySelectorAll('.table-action-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const idx = parseInt(btn.getAttribute('data-index'), 10);
        if (data[idx].final_score >= 0.6) {
          this._openDetail(data[idx]);
        }
      });
    });

    // Populate terminal log
    const logContainer = document.getElementById('terminal-log-container');
    const logEl = document.getElementById('results-terminal-log');
    if (logContainer && logEl) {
      logContainer.style.display = 'block';
      const top3 = data.slice(0, 3);
      logEl.innerHTML = `
        <div class="terminal-log-header">
          <span>System Log</span>
          <span class="material-symbols-outlined" style="font-size:16px">terminal</span>
        </div>
        <div class="log-line info">[INFO] Initializing Matrix sorting routine...</div>
        <div class="log-line info">[INFO] Applying Semantic Weight (0.65) and Keyword Match (0.35).</div>
        ${top3.some(c => !c.category.match) ? `<div class="log-line warn">[WARN] ${top3.find(c => !c.category.match)?.id || 'A candidate'} exhibits category mismatch. Applying penalty filter.</div>` : ''}
        <div class="log-line success">[SUCCESS] Matrix rendering complete. ${data.length} rows processed.</div>
        <div class="log-line cursor" style="margin-top:8px;color:var(--color-primary)">&gt; Awaiting user action_</div>
      `;
    }

    // Populate pagination
    const paginationContainer = document.getElementById('pagination-container');
    if (paginationContainer) {
      paginationContainer.style.display = 'block';
      paginationContainer.className = 'pagination';
      paginationContainer.innerHTML = `
        <span>Showing 1-${Math.min(data.length, 12)} of ${data.length} records</span>
        <div class="pagination-controls">
          <button class="pagination-btn" disabled>&lt;</button>
          <button class="pagination-btn" ${data.length <= 12 ? 'disabled' : ''}>&gt;</button>
        </div>
      `;
    }
  },

  _updateSidebar(candidate) {
    const sidebar = document.getElementById('results-sidebar');
    if (!sidebar) return;

    const weights = [
      { label: 'Semantic Weight', value: 0.65, cls: '' },
      { label: 'Keyword Match', value: 0.35, cls: 'color-primary' },
    ];

    sidebar.innerHTML = `
      <h3>Model Parameters</h3>
      ${weights.map(w => `
        <div class="param-group">
          <div class="param-value" style="display:flex;justify-content:space-between">
            <span>${w.label}</span>
            <span class="${w.cls}" style="font-weight:600">${w.value}</span>
          </div>
          <div class="param-bar">
            <div class="param-bar-fill" style="width:${Math.round(w.value * 100)}%"></div>
          </div>
        </div>
      `).join('')}
      <div class="param-group" style="margin-top:16px;padding-top:12px;border-top:1px solid var(--color-outline)">
        <div class="param-label">Penalties</div>
        <div class="param-value">Category mismatch: &times;0.50</div>
        <div class="param-value">Low confidence: &times;0.80</div>
        <div class="param-value">Missing skills: &times;0.70</div>
        <div class="param-value">Experience gap: &times;0.85</div>
      </div>
      <div class="param-group" style="margin-top:16px;padding-top:12px;border-top:1px solid var(--color-outline)">
        <div class="param-label">Thresholds</div>
        <div class="param-value">Good Fit &ge; 0.60</div>
        <div class="param-value">Potential Fit &ge; 0.35</div>
      </div>
      <div class="param-group" style="margin-top:16px;padding-top:12px;border-top:1px solid var(--color-outline)">
        <div class="param-label">Top Candidate</div>
        <div class="param-value" style="font-weight:600;color:var(--color-primary)">${candidate?.id || candidate?.candidate_id || '&mdash;'}</div>
      </div>
    `;
  },

  _renderError(container, err) {
    container.innerHTML = `
      <div class="results-empty">
        <div class="results-empty-icon" style="color:var(--color-accent)">⚠</div>
        <div class="results-empty-text" style="color:var(--color-accent)">
          Error: ${err.message || 'Failed to load results'}
        </div>
        <button class="btn btn-secondary btn-sm" onclick="Results.loadResults()">
          Retry
        </button>
      </div>
    `;
  },

  _openDetail(candidate) {
    DetailModal.open(candidate);
  },
};

// ============================================
// Inference Details — Modal
// ============================================
const DetailModal = {
  _modal: null,
  _backdrop: null,

  init() {
    this._backdrop = document.getElementById('detail-backdrop');
    this._modal = document.getElementById('detail-modal');

    // Close handlers
    document.getElementById('modal-close')?.addEventListener('click', () => this.close());
    this._backdrop?.addEventListener('click', e => {
      if (e.target === this._backdrop) this.close();
    });
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape' && this._backdrop?.classList.contains('active')) this.close();
    });
  },

  open(candidate) {
    if (!this._backdrop || !this._modal) return;
    this._candidate = candidate;
    this._renderModal(candidate);
    this._backdrop.classList.add('active');
    document.body.style.overflow = 'hidden';
  },

  close() {
    if (!this._backdrop) return;
    this._backdrop.classList.remove('active');
    document.body.style.overflow = '';
  },

  _renderModal(candidate) {
    if (!this._modal) return;

    const labelClass = candidate.label === 'Good Fit' ? 'good-fit'
      : candidate.label === 'Potential Fit' ? 'potential-fit' : 'bad-fit';

    // Mock entity extraction chain items (simulated from JD terms)
    const chainItems = [
      { jd: 'Distributed Systems', match: 'Apache Kafka' },
      { jd: 'Cloud Infrastructure', match: 'AWS ECS' },
      { jd: 'CI/CD Pipelines', match: 'GitHub Actions' },
    ];

    this._modal.innerHTML = `
      <div class="modal-header">
        <div>
          <h2>${candidate.id || candidate.candidate_id}</h2>
          <div class="modal-subtitle">Composite: ${candidate.final_score.toFixed(4)}</div>
        </div>
        <button class="modal-close" id="modal-close-inner" aria-label="Close">&cross;</button>
      </div>
      <div class="modal-body">
        <div class="modal-left">
          <div class="chart-container" id="radar-chart"></div>
          <div class="score-breakdown">
            <div class="score-metric">
              <div class="score-metric-value">${candidate.final_score.toFixed(4)}</div>
              <div class="score-metric-label">Final Score</div>
            </div>
            <div class="score-metric">
              <div class="score-metric-value">${candidate.raw_score.toFixed(4)}</div>
              <div class="score-metric-label">Raw Score</div>
            </div>
            <div class="score-metric">
              <div class="score-metric-value">${(candidate.skill_overlap * 100).toFixed(0)}%</div>
              <div class="score-metric-label">Skill Overlap</div>
            </div>
            <div class="score-metric">
              <div class="score-metric-value">${(candidate.keyword_overlap * 100).toFixed(0)}%</div>
              <div class="score-metric-label">Keyword Overlap</div>
            </div>
          </div>

          <div class="section-label">Entity Extraction</div>
          <div class="entity-chain">
            ${chainItems.map(item => `
              <div class="entity-chain-item">
                <span class="jd-label">JD:</span>
                <span class="jd-term">${item.jd}</span>
                <span class="chain-arrow material-symbols-outlined" style="font-size:14px">arrow_right_alt</span>
                <span class="candidate-match">${item.match}</span>
              </div>
            `).join('')}
          </div>

          <div class="section-label">Extracted Skills (${candidate.extracted_skills?.length || 0})</div>
          <div class="entity-chips">
            ${(candidate.extracted_skills || []).map(s =>
              `<span class="entity-chip">${s}</span>`
            ).join(' ')}
          </div>
        </div>
        <div class="modal-right">
          <div class="section-label">Score Breakdown</div>
          <div style="margin-bottom:20px">
            <div class="param-group">
              <div class="param-label">Label</div>
              <span class="label-badge ${labelClass}" style="margin-top:4px">${candidate.label}</span>
            </div>
          </div>

          <div class="section-label">Category Validation</div>
          <div style="margin-bottom:20px">
            <div class="param-group">
              <div class="param-label">Predicted</div>
              <div class="param-value" style="font-weight:600">${candidate.category.predicted}</div>
            </div>
            <div class="param-group">
              <div class="param-label">Match</div>
              <div class="param-value" style="color:${candidate.category.match ? 'var(--color-success)' : 'var(--color-accent)'}">${candidate.category.match ? '&check; Matched' : '&cross; Mismatch'}</div>
            </div>
            <div class="param-group">
              <div class="param-label">Confidence</div>
              <div class="param-value">${(candidate.category.confidence * 100).toFixed(1)}%</div>
            </div>
          </div>

          <div class="section-label">Penalty Log</div>
          <div class="penalty-block" id="penalty-log-modal">
            ${(candidate.penalties || []).length > 0 ? candidate.penalties.map(p => `
              <div class="penalty-line">
                <span>${p.value < 0 ? '[-]' : '[+]'} ${p.type}</span>
                <span class="penalty-negative">${(p.value < 0 ? '' : '+')}${p.value.toFixed(3)}</span>
              </div>
            `).join('') : '<div style="color:var(--color-outline)">No penalties applied</div>'}
          </div>

          <div class="section-label">Label Probabilities</div>
          <div>
            ${Object.entries(candidate.label_probabilities || {}).map(([label, prob]) => `
              <div class="param-group">
                <div class="param-label">${label}</div>
                <div class="param-value">${(prob * 100).toFixed(1)}%</div>
                <div class="param-bar">
                  <div class="param-bar-fill" style="width:${Math.round(prob * 100)}%;background:${label === 'Good Fit' ? 'var(--color-success)' : label === 'Bad Fit' ? 'var(--color-accent)' : 'var(--color-warning)'}"></div>
                </div>
              </div>
            `).join('')}
          </div>
        </div>
      </div>
    `;

    // Wire inner close button
    document.getElementById('modal-close-inner')?.addEventListener('click', () => this.close());

    // Draw radar chart
    const radarData = candidate.radar || {
      sem: 0.8, kw: 0.6, exp: 0.7, edu: 0.9, keyw: 0.5
    };
    this._drawRadarChart('radar-chart', radarData);
  },

  _drawRadarChart(containerId, data) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const size = Math.min(container.clientWidth, container.clientHeight, 400);
    const cx = size / 2;
    const cy = size / 2;
    const radius = size * 0.35;

    const labels = ['Semantic', 'Syntax', 'Experience', 'Education', 'Keywords'];
    const values = [data.sem, data.kw, data.exp, data.edu, data.keyw];
    const numAxes = labels.length;
    const angleStep = (2 * Math.PI) / numAxes;

    // Start from top (-π/2)
    const getPoint = (index, value) => {
      const angle = -Math.PI / 2 + index * angleStep;
      return {
        x: cx + radius * value * Math.cos(angle),
        y: cy + radius * value * Math.sin(angle),
      };
    };

    const getAxisEnd = (index) => {
      const angle = -Math.PI / 2 + index * angleStep;
      return {
        x: cx + radius * 1.05 * Math.cos(angle),
        y: cy + radius * 1.05 * Math.sin(angle),
      };
    };

    // Build SVG
    let svg = `<svg viewBox="0 0 ${size} ${size}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Radar chart of candidate scores">`;

    // Grid rings
    for (let ring = 1; ring <= 3; ring++) {
      const r = (radius / 3) * ring;
      let path = '';
      for (let i = 0; i <= numAxes; i++) {
        const angle = -Math.PI / 2 + (i % numAxes) * angleStep;
        const px = cx + r * Math.cos(angle);
        const py = cy + r * Math.sin(angle);
        path += i === 0 ? `M ${px} ${py}` : `L ${px} ${py}`;
      }
      svg += `<path d="${path}" fill="none" stroke="#ddd" stroke-width="1"/>`;
    }

    // Axis lines
    for (let i = 0; i < numAxes; i++) {
      const end = getAxisEnd(i);
      svg += `<line x1="${cx}" y1="${cy}" x2="${end.x}" y2="${end.y}" stroke="#ddd" stroke-width="1"/>`;

      // Labels
      const labelAngle = -Math.PI / 2 + i * angleStep;
      const lx = cx + (radius * 1.2) * Math.cos(labelAngle);
      const ly = cy + (radius * 1.2) * Math.sin(labelAngle);
      const anchor = labelAngle > -0.3 && labelAngle < 0.3 ? 'middle'
        : labelAngle > 0.3 && labelAngle < 2.5 ? 'start' : 'end';
      svg += `<text x="${lx}" y="${ly}" text-anchor="${anchor}" dominant-baseline="middle" font-family="IBM Plex Mono, monospace" font-size="11" fill="#666">${labels[i]}</text>`;
    }

    // Data polygon (fill)
    let polyPath = '';
    for (let i = 0; i <= numAxes; i++) {
      const p = getPoint(i % numAxes, values[i % numAxes]);
      polyPath += i === 0 ? `M ${p.x} ${p.y}` : `L ${p.x} ${p.y}`;
    }
    svg += `<path d="${polyPath}" fill="rgba(0,0,255,0.15)" stroke="#0000ff" stroke-width="2"/>`;

    // Data points
    for (let i = 0; i < numAxes; i++) {
      const p = getPoint(i, values[i]);
      svg += `<circle cx="${p.x}" cy="${p.y}" r="4" fill="#0000ff" stroke="#fff" stroke-width="2"/>`;
    }

    // Center label
    svg += `<text x="${cx}" y="${cy}" text-anchor="middle" dominant-baseline="middle" font-family="IBM Plex Mono, monospace" font-size="12" font-weight="600" fill="#111">Score</text>`;

    svg += '</svg>';
    container.innerHTML = svg;
  },
};

// ============================================
// Methodology — TOC
// ============================================
const Methodology = {
  _tocLinks: null,
  _sections: null,

  init() {
    this._tocLinks = document.querySelectorAll('.toc-list a');
    this._sections = document.querySelectorAll('.methodology-content section');

    this._tocLinks.forEach(link => {
      link.addEventListener('click', e => {
        e.preventDefault();
        const target = link.getAttribute('href');
        const el = document.querySelector(target);
        if (el) {
          const offset = 80;
          const top = el.getBoundingClientRect().top + window.scrollY - offset;
          window.scrollTo({ top, behavior: 'smooth' });
        }
      });
    });

    // Scroll spy
    window.addEventListener('scroll', () => this._updateActiveTOC());
  },

  _updateActiveTOC() {
    if (!this._tocLinks.length) return;
    let current = '';
    const scrollPos = window.scrollY + 100;

    this._sections.forEach(section => {
      const top = section.offsetTop;
      const height = section.offsetHeight;
      if (scrollPos >= top && scrollPos < top + height) {
        current = '#' + section.id;
      }
    });

    this._tocLinks.forEach(link => {
      link.classList.toggle('active', link.getAttribute('href') === current);
    });
  },
};

// ============================================
// Navigation — Active Page Tracking
// ============================================
function initNavTracking() {
  const observer = new IntersectionObserver((entries) => {
    // We use the router's hash-based tracking instead
  });
}

// ============================================
// Copy button for code blocks
// ============================================
function initCodeCopy() {
  document.querySelectorAll('.code-block').forEach(block => {
    const btn = block.querySelector('.copy-btn');
    if (!btn) return;

    btn.addEventListener('click', async () => {
      const code = block.querySelector('code')?.textContent || '';
      try {
        await navigator.clipboard.writeText(code);
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = '[Copy]'; }, 2000);
      } catch {
        btn.textContent = 'Failed';
        setTimeout(() => { btn.textContent = '[Copy]'; }, 2000);
      }
    });
  });
}

// ============================================
// Execute Pipeline
// ============================================
async function executePipeline() {
  const jdText = document.getElementById('jd-text')?.value?.trim();
  const fileCount = Sandbox.getFileCount();

  if (!jdText && fileCount === 0) {
    alert('Please paste a job description or upload resumes.');
    return;
  }

  await Terminal.simulateProcessing();
}

// ============================================
// Scroll Reveal — Intersection Observer
// ============================================
function initScrollReveal() {
  const els = document.querySelectorAll('.reveal');
  if (!els.length) return;

  const observer = new IntersectionObserver(
    entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.15, rootMargin: '0px 0px -40px 0px' }
  );

  els.forEach(el => observer.observe(el));
}

// Animate score bars after results render
function animateScoreBars() {
  const bars = document.querySelectorAll('.score-bar');
  bars.forEach(bar => {
    // Re-trigger the CSS animation by reflow
    bar.style.animation = 'none';
    void bar.offsetHeight;
    bar.style.animation = '';
  });
}

// ============================================
// Init
// ============================================
document.addEventListener('DOMContentLoaded', () => {
  Router.init();
  Terminal.init();
  Sandbox.init();
  DetailModal.init();
  Methodology.init();
  initCodeCopy();

  // Wire execute button
  document.getElementById('btn-execute')?.addEventListener('click', executePipeline);

  // Wire reset button
  document.getElementById('btn-reset')?.addEventListener('click', () => {
    Sandbox.clearFiles();
    Terminal.reset();
    document.getElementById('jd-text').value = '';
    document.getElementById('file-count').textContent = '0 files';
  });

  // Init scroll reveal (hero entrance handled by CSS)
  initScrollReveal();

  // Re-init scroll reveal after page changes (new .reveal elements appear)
  Router._onPageChange = () => {
    setTimeout(initScrollReveal, 100);
  };

  console.log('The Lab — TriadRank ATS Showcase initialized');
  console.log(`API mode: ${CONFIG.useMockData ? 'MOCK DATA' : 'LIVE API @ ' + CONFIG.apiBaseUrl}`);
});
