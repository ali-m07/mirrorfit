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

function jalaliToGregorian(jy, jm, jd) {
  let gy = jy <= 979 ? 621 : 1600;
  jy -= jy <= 979 ? 0 : 979;
  const days = 365 * jy + Math.floor(jy / 33) * 8 + Math.floor((jy % 33 + 3) / 4)
    + 78 + (jm < 7 ? (jm - 1) * 31 : (jm - 7) * 30 + 186) + jd - 1;
  gy += 400 * Math.floor(days / 146097);
  let d = days % 146097;
  if (d >= 36525) {
    gy += 100 * Math.floor((d - 1) / 36525);
    d = (d - 1) % 36525;
  }
  gy += 4 * Math.floor(d / 1461);
  d %= 1461;
  if (d >= 366) {
    gy += Math.floor((d - 1) / 365);
    d = (d - 1) % 365;
  }
  const sal_a = [0, 31, (gy % 4 === 0 && gy % 100 !== 0) || gy % 400 === 0 ? 29 : 28,
    31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  let gm = 0;
  while (gm < 13 && d >= sal_a[gm]) {
    d -= sal_a[gm++];
  }
  return [gy, gm, d + 1];
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

function formatJalaliDate(jy, jm, jd) {
  return `${jy}/${String(jm).padStart(2, '0')}/${String(jd).padStart(2, '0')}`;
}

function jalaliMonthName(jm) {
  return PERSIAN_MONTHS[jm - 1] || '';
}

function isDateBeforeOrEqual(jy1, jm1, jd1, jy2, jm2, jd2) {
  if (jy1 !== jy2) return jy1 < jy2;
  if (jm1 !== jm2) return jm1 < jm2;
  return jd1 <= jd2;
}

function isTempExpenseActive(expense, targetYear, targetMonth) {
  if (!expense.endDate) return true;
  const end = parseJalaliDate(expense.endDate);
  if (!end) return true;
  const [ey, em, ed] = end;
  if (targetYear > ey) return false;
  if (targetYear < ey) return true;
  if (targetMonth > em) return false;
  if (targetMonth < em) return true;
  return true;
}

function calculateMonthBudget(data, targetYear, targetMonth) {
  const totalIncome = data.incomes.reduce((sum, i) => sum + parseAmount(i.amount), 0);

  const fixedTotal = data.fixedExpenses.reduce((sum, e) => sum + parseAmount(e.amount), 0);

  const tempTotal = data.tempExpenses
    .filter(e => isTempExpenseActive(e, targetYear, targetMonth))
    .reduce((sum, e) => sum + parseAmount(e.amount), 0);

  const [todayY, todayM] = getTodayJalali();
  const isCurrentMonth = targetYear === todayY && targetMonth === todayM;

  const oneTimeTotal = isCurrentMonth
    ? data.oneTimeExpenses.reduce((sum, e) => sum + parseAmount(e.amount), 0)
    : 0;

  const totalExpenses = fixedTotal + tempTotal + oneTimeTotal;
  const available = totalIncome - totalExpenses;

  return {
    year: targetYear,
    month: targetMonth,
    monthName: jalaliMonthName(targetMonth),
    totalIncome,
    fixedTotal,
    tempTotal,
    oneTimeTotal,
    totalExpenses,
    available,
    isCurrentMonth,
    expenseBreakdown: [
      ...data.fixedExpenses.map(e => ({ name: e.name, amount: parseAmount(e.amount), type: 'ثابت' })),
      ...data.tempExpenses
        .filter(e => isTempExpenseActive(e, targetYear, targetMonth))
        .map(e => ({ name: e.name, amount: parseAmount(e.amount), type: 'موقت' })),
      ...(isCurrentMonth ? data.oneTimeExpenses.map(e => ({ name: e.name, amount: parseAmount(e.amount), type: 'یک‌باره' })) : [])
    ]
  };
}

function getForecast(data, monthsAhead = 6) {
  const [startY, startM] = getTodayJalali();
  const forecast = [];

  for (let i = 0; i < monthsAhead; i++) {
    let y = startY;
    let m = startM + i;
    while (m > 12) {
      m -= 12;
      y += 1;
    }
    forecast.push(calculateMonthBudget(data, y, m));
  }
  return forecast;
}

function getBudgetStatus(available, totalIncome) {
  const ratio = totalIncome > 0 ? available / totalIncome : 0;
  if (available < 0) return { label: 'کسری!', class: 'bad' };
  if (ratio < 0.15) return { label: 'تنگ', class: 'tight' };
  return { label: 'مناسب', class: 'good' };
}

function getDailyBudget(available) {
  const now = new Date();
  const [jy, jm, jd] = getTodayJalali();
  const daysInMonth = jm <= 6 ? 31 : (jm <= 11 ? 30 : (isLeapJalali(jy) ? 30 : 29));
  const remainingDays = Math.max(1, daysInMonth - jd + 1);
  return available / remainingDays;
}

function isLeapJalali(jy) {
  const breaks = [1, 5, 9, 13, 17, 22, 26, 30];
  const jp = jy - (jy >= 0 ? 474 : 473);
  const jy2 = 474 + (jp % 2820);
  return breaks.includes((jy2 + 38) * 682 % 2816);
}
