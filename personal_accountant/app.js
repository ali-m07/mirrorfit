const STORAGE_KEY = 'personal_accountant_data_v5';

const DEFAULT_DATA = {
  planningMonth: { year: 1405, month: 6 },
  incomes: [
    { id: 'inc1', name: 'دریافتی خالص (حقوق مرداد)', amount: 64000000, forMonth: '1405/06' }
  ],
  fixedExpenses: [
    { id: 'fix1', name: 'دیجی‌پی', amount: 9500000 },
    { id: 'fix2', name: 'اسنپ', amount: 6000000 },
    { id: 'fix3', name: 'وام ملت', amount: 7100000 }
  ],
  tempExpenses: [
    { id: 'tmp1', name: 'رسالت', amount: 15000000, endDate: '1405/07/29' },
    { id: 'tmp2', name: 'قسط شهریور', amount: 2100000, endDate: '1405/06/05' },
    { id: 'tmp3', name: 'قسط آذر', amount: 2800000, endDate: '1405/09/10' },
    { id: 'tmp4', name: 'قسط شرکت', amount: 6500000, endDate: '1405/12/29' }
  ],
  oneTimeExpenses: [
    { id: 'ot1', name: 'بدهی اضافی', amount: 18000000, targetMonth: '1405/06' }
  ]
};

let appData = loadData();

function loadData() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) return JSON.parse(saved);
  } catch (_) { /* ignore */ }
  return JSON.parse(JSON.stringify(DEFAULT_DATA));
}

function saveData() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(appData));
  render();
}

function uid() {
  return 'id_' + Date.now() + '_' + Math.random().toString(36).slice(2, 7);
}

function addIncome() {
  const [py, pm] = getPlanningMonth(appData);
  appData.incomes.push({ id: uid(), name: 'دریافتی', amount: 0, forMonth: `${py}/${String(pm).padStart(2, '0')}` });
  saveData();
}

function addFixedExpense() {
  appData.fixedExpenses.push({ id: uid(), name: 'قسط جدید', amount: 0 });
  saveData();
}

function addTempExpense() {
  appData.tempExpenses.push({ id: uid(), name: 'قسط موقت', amount: 0, endDate: '1405/12/29' });
  saveData();
}

function addOneTimeExpense() {
  const [py, pm] = getPlanningMonth(appData);
  appData.oneTimeExpenses.push({ id: uid(), name: 'هزینه', amount: 0, targetMonth: `${py}/${String(pm).padStart(2, '0')}` });
  saveData();
}

function deleteItem(list, id) {
  appData[list] = appData[list].filter(item => item.id !== id);
  saveData();
}

function updateItem(list, id, field, value) {
  const item = appData[list].find(i => i.id === id);
  if (item) {
    item[field] = value;
    saveData();
  }
}

function renderItemList(containerId, items, listName, fields) {
  const container = document.getElementById(containerId);
  container.innerHTML = items.map(item => {
    const inputs = fields.map(f => {
      if (f.type === 'amount') {
        return `<input class="amount ${listName === 'incomes' ? 'income-color' : ''}" type="text"
          value="${parseAmount(item[f.key]).toLocaleString('en-US')}"
          onchange="updateItem('${listName}','${item.id}','${f.key}',parseAmount(this.value)); this.value=parseAmount(this.value).toLocaleString('en-US')"
          placeholder="مبلغ">`;
      }
      if (f.type === 'date') {
        return `<input class="date-input" type="text" value="${item[f.key] || ''}"
          onchange="updateItem('${listName}','${item.id}','${f.key}',this.value)"
          placeholder="1405/07/29">`;
      }
      if (f.type === 'month') {
        return `<input class="date-input" type="text" value="${item[f.key] || ''}"
          onchange="updateItem('${listName}','${item.id}','${f.key}',this.value)"
          placeholder="1405/06">`;
      }
      return `<input type="text" value="${item[f.key]}"
        onchange="updateItem('${listName}','${item.id}','${f.key}',this.value)"
        placeholder="نام">`;
    }).join('');
    return `<div class="item-row">${inputs}<button class="btn-delete" onclick="deleteItem('${listName}','${item.id}')">✕</button></div>`;
  }).join('');
}

function renderSummary(b) {
  const el = document.getElementById('summaryCards');
  const cls = b.remaining >= 0 ? 'positive' : 'negative';
  el.innerHTML = `
    <div class="card">
      <div class="card-label">دریافتی خالص</div>
      <div class="card-value income">${formatMoney(b.netReceived)}</div>
      <div class="card-sub">همین مبلغ دستته — ثابت</div>
    </div>
    <div class="card">
      <div class="card-label">جمع خرج‌های ${b.monthName}</div>
      <div class="card-value expense">${formatMoney(b.totalExpenses)}</div>
      <div class="card-sub">${b.expenseBreakdown.length} مورد</div>
    </div>
    <div class="card card-main">
      <div class="card-label">${b.remaining >= 0 ? 'مانده' : 'کسری'}</div>
      <div class="card-value balance ${cls}">${formatMoney(b.remaining)}</div>
      <div class="card-sub">${b.remaining >= 0 ? 'برای زندگی در ' + b.monthName : formatMoney(Math.abs(b.remaining)) + ' کم داری'}</div>
    </div>
    <div class="card">
      <div class="card-label">بودجه روزانه</div>
      <div class="card-value ${cls}">${formatMoney(getDailyBudget(b.remaining, b.year, b.month))}</div>
      <div class="card-sub">${b.remaining >= 0 ? 'قابل خرج' : 'در صورت تأمین کسری'}</div>
    </div>
  `;
}

function renderAlert(b) {
  const box = document.getElementById('alertBox');
  if (b.remaining < 0) {
    box.innerHTML = `<div class="alert danger">
      <strong>🔴 ${formatMoney(Math.abs(b.remaining))} منفی هستی</strong><br>
      ${formatMoney(b.netReceived)} داری − ${formatMoney(b.totalExpenses)} خرج = ${formatMoney(b.remaining)}
    </div>`;
  } else if (b.remaining / b.netReceived < 0.15) {
    box.innerHTML = `<div class="alert warning">
      <strong>⚡ ${formatMoney(b.remaining)} مونده</strong> — خیلی کم، محتاط باش.
    </div>`;
  } else {
    box.innerHTML = `<div class="alert success">
      <strong>✅ ${formatMoney(b.remaining)} مونده</strong> — ${formatMoney(getDailyBudget(b.remaining, b.year, b.month))} در روز.
    </div>`;
  }
}

function renderActions(b, forecast) {
  const tips = getActionAdvice(b, forecast);
  document.getElementById('actionBox').innerHTML = tips.map(t =>
    `<div class="action-item">${t}</div>`
  ).join('');
}

function renderForecast(forecast) {
  const maxVal = Math.max(...forecast.map(f => f.netReceived), 1);
  document.getElementById('forecastChart').innerHTML = forecast.map(f => {
    const incH = (f.netReceived / maxVal) * 140;
    const expH = (f.totalExpenses / maxVal) * 140;
    const remH = (Math.max(f.remaining, 0) / maxVal) * 140;
    return `<div class="bar-group">
      <div class="bar-container">
        <div class="bar income-bar" style="height:${incH}px"></div>
        <div class="bar expense-bar" style="height:${expH}px"></div>
        <div class="bar balance-bar" style="height:${remH}px"></div>
      </div>
      <div class="bar-label">${f.monthName}${f.isPlanningMonth ? ' ★' : ''}</div>
    </div>`;
  }).join('');

  document.getElementById('forecastBody').innerHTML = forecast.map(f => {
    const st = getBudgetStatus(f.remaining, f.netReceived);
    return `<tr${f.isPlanningMonth ? ' class="planning-row"' : ''}>
      <td>${f.monthName}${f.isPlanningMonth ? ' ← الان' : ''}</td>
      <td class="amount">${formatMoney(f.netReceived)}</td>
      <td class="amount">${formatMoney(f.totalExpenses)}</td>
      <td class="amount" ${f.remaining < 0 ? 'style="color:var(--danger)"' : ''}>${formatMoney(f.remaining)}</td>
      <td><span class="status-badge ${st.class}">${st.label}</span></td>
    </tr>`;
  }).join('');
}

function renderBreakdown(b) {
  const rows = b.expenseBreakdown.map(e =>
    `<div class="budget-row deduct">
      <span class="label">− ${e.name} <small>(${e.type})</small></span>
      <span class="value expense-val">${formatMoney(e.amount)}</span>
    </div>`
  ).join('');

  document.getElementById('budgetBreakdown').innerHTML = `
    <div class="budget-row">
      <span class="label">دریافتی خالص</span>
      <span class="value positive">${formatMoney(b.netReceived)}</span>
    </div>
    ${rows}
    <div class="budget-row subtotal">
      <span class="label">= جمع خرج‌ها</span>
      <span class="value expense-val">${formatMoney(b.totalExpenses)}</span>
    </div>
    <div class="budget-row highlight final">
      <span class="label">${b.remaining >= 0 ? '✅ مانده' : '🔴 کسری'}</span>
      <span class="value ${b.remaining >= 0 ? 'positive' : 'negative'}">${formatMoney(b.remaining)}</span>
    </div>
  `;
}

function renderDate() {
  const [jy, jm, jd] = getTodayJalali();
  document.getElementById('currentDate').textContent = `امروز: ${jd} ${jalaliMonthName(jm)} ${jy}`;
}

function render() {
  renderDate();
  renderItemList('incomeList', appData.incomes, 'incomes', [
    { key: 'name', type: 'text' }, { key: 'amount', type: 'amount' }, { key: 'forMonth', type: 'month' }
  ]);
  renderItemList('fixedExpenseList', appData.fixedExpenses, 'fixedExpenses', [
    { key: 'name', type: 'text' }, { key: 'amount', type: 'amount' }
  ]);
  renderItemList('tempExpenseList', appData.tempExpenses, 'tempExpenses', [
    { key: 'name', type: 'text' }, { key: 'amount', type: 'amount' }, { key: 'endDate', type: 'date' }
  ]);
  renderItemList('oneTimeList', appData.oneTimeExpenses, 'oneTimeExpenses', [
    { key: 'name', type: 'text' }, { key: 'amount', type: 'amount' }, { key: 'targetMonth', type: 'month' }
  ]);

  const [py, pm] = getPlanningMonth(appData);
  const budget = calculateMonthBudget(appData, py, pm);
  renderSummary(budget);
  renderAlert(budget);
  renderBreakdown(budget);
  renderActions(budget);
  renderForecast(getForecast(appData, 6));
}

render();
