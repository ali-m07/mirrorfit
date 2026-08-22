const PERSIAN_MONTHS = [
  'فروردین', 'اردیبهشت', 'خرداد', 'تیر', 'مرداد', 'شهریور',
  'مهر', 'آبان', 'آذر', 'دی', 'بهمن', 'اسفند'
];

function formatMoney(amount) {
  const abs = Math.abs(Math.round(amount));
  return abs.toLocaleString('fa-IR') + ' تومان';
}

function parseAmount(str) {
  if (typeof str === 'number') return str;
  const cleaned = String(str).replace(/[^\d.]/g, '');
  return parseFloat(cleaned) || 0;
}

function gregorianToJalali(gy, gm, gd) {
  const g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
  let jy = gy <= 1600 ? 0 : 979;
  gy -= gy <= 1600 ? 621 : 1600;
  const gy2 = gm > 2 ? gy + 1 : gy;
  let days = 365 * gy + Math.floor((gy2 + 3) / 4) - Math.floor((gy2 + 99) / 100)
    + Math.floor((gy2 + 399) / 400) - 80 + gd + g_d_m[gm - 1];
  jy += 33 * Math.floor(days / 12053);
  days %= 12053;
  jy += 4 * Math.floor(days / 1461);
  days %= 1461;
  jy += Math.floor((days - 1) / 365);
  if (days > 365) days = (days - 1) % 365;
  const jm = days < 186 ? 1 + Math.floor(days / 31) : 7 + Math.floor((days - 186) / 30);
  const jd = 1 + (days < 186 ? days % 31 : (days - 186) % 30);
  return [jy, jm, jd];
}

function getTodayJalali() {
  const now = new Date();
  return gregorianToJalali(now.getFullYear(), now.getMonth() + 1, now.getDate());
}

function parseJalaliDate(str) {
  const parts = String(str).replace(/-/g, '/').split('/');
  if (parts.length !== 3) return null;
  return [parseInt(parts[0], 10), parseInt(parts[1], 10), parseInt(parts[2], 10)];
}

function jalaliMonthName(jm) {
  return PERSIAN_MONTHS[jm - 1] || '';
}

function parseTargetMonth(str) {
  if (!str) return null;
  const parts = String(str).replace(/-/g, '/').split('/');
  if (parts.length < 2) return null;
  return [parseInt(parts[0], 10), parseInt(parts[1], 10)];
}

function matchesTargetMonth(item, targetYear, targetMonth) {
  const tm = parseTargetMonth(item.targetMonth || item.forMonth);
  if (!tm) return true;
  return tm[0] === targetYear && tm[1] === targetMonth;
}

function isTempExpenseActive(expense, targetYear, targetMonth) {
  if (!expense.endDate) return true;
  const end = parseJalaliDate(expense.endDate);
  if (!end) return true;
  const [ey, em] = end;
  if (targetYear > ey) return false;
  if (targetYear < ey) return true;
  if (targetMonth > em) return false;
  return true;
}

function getNetIncome(data, targetYear, targetMonth) {
  const matched = data.incomes.filter(i => matchesTargetMonth(i, targetYear, targetMonth));
  if (matched.length) {
    return matched.reduce((sum, i) => sum + parseAmount(i.amount), 0);
  }
  return data.incomes.reduce((sum, i) => sum + parseAmount(i.amount), 0);
}

function calculateMonthBudget(data, targetYear, targetMonth) {
  const netReceived = getNetIncome(data, targetYear, targetMonth);

  const expenseBreakdown = [
    ...data.fixedExpenses.map(e => ({
      name: e.name, amount: parseAmount(e.amount), type: 'ثابت'
    })),
    ...data.tempExpenses
      .filter(e => isTempExpenseActive(e, targetYear, targetMonth))
      .map(e => ({ name: e.name, amount: parseAmount(e.amount), type: 'موقت' })),
    ...data.oneTimeExpenses
      .filter(e => matchesTargetMonth(e, targetYear, targetMonth))
      .map(e => ({ name: e.name, amount: parseAmount(e.amount), type: 'یک‌باره' }))
  ];

  const fixedTotal = expenseBreakdown.filter(e => e.type === 'ثابت').reduce((s, e) => s + e.amount, 0);
  const tempTotal = expenseBreakdown.filter(e => e.type === 'موقت').reduce((s, e) => s + e.amount, 0);
  const oneTimeTotal = expenseBreakdown.filter(e => e.type === 'یک‌باره').reduce((s, e) => s + e.amount, 0);
  const totalExpenses = fixedTotal + tempTotal + oneTimeTotal;
  const remaining = netReceived - totalExpenses;

  const [todayY, todayM] = getTodayJalali();
  const isPlanningMonth = data.planningMonth
    && data.planningMonth.year === targetYear
    && data.planningMonth.month === targetMonth;

  return {
    year: targetYear,
    month: targetMonth,
    monthName: jalaliMonthName(targetMonth),
    netReceived,
    fixedTotal,
    tempTotal,
    oneTimeTotal,
    totalExpenses,
    remaining,
    isPlanningMonth,
    expenseBreakdown
  };
}

function getPlanningMonth(data) {
  if (data.planningMonth) {
    return [data.planningMonth.year, data.planningMonth.month];
  }
  const [y, m] = getTodayJalali();
  let nm = m + 1, ny = y;
  if (nm > 12) { nm = 1; ny += 1; }
  return [ny, nm];
}

function getForecast(data, monthsAhead = 6) {
  const [startY, startM] = getPlanningMonth(data);
  const forecast = [];
  for (let i = 0; i < monthsAhead; i++) {
    let y = startY, m = startM + i;
    while (m > 12) { m -= 12; y += 1; }
    forecast.push(calculateMonthBudget(data, y, m));
  }
  return forecast;
}

function getBudgetStatus(remaining, netReceived) {
  if (remaining < 0) return { label: 'منفی', class: 'bad' };
  if (netReceived > 0 && remaining / netReceived < 0.15) return { label: 'تنگ', class: 'tight' };
  return { label: 'مثبت', class: 'good' };
}

function getDailyBudget(remaining, targetYear, targetMonth) {
  const daysInMonth = targetMonth <= 6 ? 31 : (targetMonth <= 11 ? 30 : 29);
  return remaining / daysInMonth;
}

function getActionAdvice(budget, forecast) {
  const { remaining, netReceived, totalExpenses, expenseBreakdown, monthName } = budget;
  const tips = [];

  if (remaining >= 0) {
    tips.push(`✅ ${formatMoney(remaining)} مونده — برای ${monthName} روزانه ~${formatMoney(getDailyBudget(remaining, budget.year, budget.month))} می‌تونی خرج کنی.`);
    if (remaining / netReceived < 0.2) {
      tips.push('⚡ بودجه تنگه — خرج‌های غیرضروری رو حذف کن.');
    }
    return tips;
  }

  const deficit = Math.abs(remaining);
  tips.push(`🔴 ${formatMoney(deficit)} منفی هستی.`);
  tips.push(`${formatMoney(netReceived)} − ${formatMoney(totalExpenses)} = ${formatMoney(remaining)}`);

  const sorted = [...expenseBreakdown].sort((a, b) => b.amount - a.amount);
  if (sorted.length) {
    tips.push(`بزرگ‌ترین خرج‌ها: ${sorted.slice(0, 3).map(e => `${e.name} (${formatMoney(e.amount)})`).join('، ')}`);
  }

  tips.push('راه‌حل‌های ممکن:');
  tips.push(`• تأمین ${formatMoney(deficit)} از پس‌انداز، خانواده، یا قرض موقت`);
  tips.push('• مذاکره برای به تأخیر انداختن یک قسط بزرگ (مثلاً رسالت یا بدهی ۱۸M)');
  tips.push('• به تعویق انداختن قسط شرکت (۶.۵M) اگه امکانش هست');

  if (forecast && forecast.length > 1) {
    const next = forecast[1];
    if (next.remaining > remaining) {
      tips.push(`• از ${next.monthName} وضعیت بهتر می‌شه — ${formatMoney(next.remaining)} مانده (${formatMoney(next.remaining - remaining)} بهتر)`);
    }
  }

  return tips;
}
