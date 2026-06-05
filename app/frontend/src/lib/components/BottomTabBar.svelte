<script lang="ts">
  import { page } from '$app/state';
  import IconHome from '~icons/mdi/home-variant';
  import IconInstances from '~icons/mdi/file-document-multiple-outline';
  import IconSections from '~icons/mdi/account-group-outline';

  const tabs = [
    { id: 'home',      href: '/',          icon: IconHome,      label: 'Home'      },
    { id: 'instances', href: '/instances',  icon: IconInstances, label: 'Instances' },
    { id: 'sections',  href: '/sections',   icon: IconSections,  label: 'Sections'  },
  ] as const;

  function isActive(href: string): boolean {
    if (href === '/') return page.url.pathname === '/';
    return page.url.pathname.startsWith(href);
  }
</script>

<nav class="bg-card shrink-0
  border-t border-border h-[60px] flex flex-row
  md:border-t-0 md:border-r md:h-full md:w-16 md:flex-col md:py-4
">
  {#each tabs as tab}
    <a
      href={tab.href}
      class="flex-1 flex flex-col items-center justify-center gap-[3px] no-underline
             md:flex-none md:w-full md:h-[60px] md:gap-1"
    >
      <tab.icon
        class="size-[22px] transition-colors {isActive(tab.href) ? 'text-primary-700' : 'text-foreground/55'}"
      />
      <span
        class="text-[10px] font-semibold tracking-[0.02em] transition-colors {isActive(tab.href) ? 'text-primary-700' : 'text-foreground/55'}"
      >
        {tab.label}
      </span>
    </a>
  {/each}
</nav>
