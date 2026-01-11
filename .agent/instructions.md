Bezwzględnie stosuj się do poniższych instrukcji:

Jesteśmy w folderze w którym piszę kod do obliczeń przekrojów czynnych wzbudzenia i jonizacji na podstawie algorytmu DWBA opisanego w pliku article.md. Informacje o poszczególnych modułach są zawarte jako komentarze oraz w dokumentacji README.md, zmiany zapisywane są w pliku CHANGELOG.md. Do kodu wprowadzono dużą ilość dodatków i usprawnień, które oczywiście są tam celowo, w tym moduł jonizacji wprowadzony z pomocą dodatkowej literatury, którą również weź pod uwagę: 
- D. Bote, F. Salvat, "Calculations of inner-shell ionization by electron impact", Phys. Rev. A (2008).
- S. Jones, D. H. Madison, "Three-body distorted-wave Born approximation for electron-atom ionization", Phys. Rev. A (2003)
- Llovet, Powell, Salvat, Jabłoński (2014) – „Cross Sections for Inner-Shell Ionization by Electron Impact” (J. Phys. Chem. Ref. Data)

Zawsze przed rozpoczęciem jakiejkolwiek pracy:
- Zapoznaj się proszę zarówno z artykułem, z którego wyciągnij jak najwięcej wiedzy, ogólnymi informacjami na temat DWBA do obliczeń przekrojów czynnych oraz z wszystkimi plikami wykonawczymi. 
- Przeanalizuj pełną literaturę i zdobądź jak najwięcej wiedzy na temat DWBA.
- Przeanalizuj dokumentację w tym README.md oraz CHANGELOG.md.

Podczas pracy zawsze:
- Pamiętaj że chcemy jak najlepszej implementacji, jakości kodu i optymalizacji. 
- Aktualizuj dokumentację i README.md, sprawdzaj aktualny commit i aktualizuj CHANGELOG.md. Pamiętaj o zmianie plików konfiguracyjnych oraz ich dokumentacji.
- Weryfikuj swoją pracę.
- Każda sensowna zmiana → nowy branch → PR → merge.
- Przygotowuj szczegółowy plan implementacji zmian i poprawek

Cechuj się:
- Proponuj różne rozwiązania.
- Dbałość o stabilność numeryczną: Zwracaj szczególną uwagę na błędy zaokrągleń, obsługę osobliwości w całkach radialnych oraz stabilność rozwiązań równań różniczkowych (np. Schrödingera czy Diraca).
- Optymalizacja wydajnościowa: Stosuj wektoryzację (NumPy/SciPy), unikaj redundantnych obliczeń w pętlach i rozważ zrównoleglanie procesów tam, gdzie to możliwe (multiprocessing/numba), korzystaj z CuPy.
- Modułowość i skalowalność: Projektuj kod zgodnie z zasadami SOLID, oddzielając warstwę fizyczną (potencjały, funkcje falowe) od silnika numerycznego (integratory, solvery).
- Rygorystyczna weryfikacja: Każda zmiana w algorytmie musi być poparta testami jednostkowymi oraz porównaniem wyników z danymi tabelarycznymi z literatury (np. bazy NIST lub prace Salvata).
- Dokumentacja matematyczna: W docstringach używaj notacji LaTeX do opisu implementowanych wzorów, aby zapewnić pełną zgodność z artykułami źródłowymi.
- Zarządzanie błędami: Implementuj szczegółowe logowanie procesów iteracyjnych i sprawdzaj fizyczną poprawność wyników (np. zachowanie energii, ortogonalność funkcji falowych).
- Adaptacyjność: Bądź gotowy na refaktoryzację istniejących modułów, jeśli nowa literatura sugeruje bardziej precyzyjne podejście do potencjałów wymiennych lub korelacyjnych.
- Respektuj dokumentację i strukturę projektu, skrypty diagnostyczne zapisuj w folderze debug, korzystaj z istniejących skryptów diagnostycznych i ulepszaj je nie usuwając zawartości aby mogły się przydać w przyszłości.
